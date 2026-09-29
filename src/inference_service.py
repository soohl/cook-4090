"""Host inference boundary. Docker reaches this service through a Unix socket."""

import asyncio
import base64
import fcntl
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import time
import tempfile
import socket as sockets

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from . import images
from .manager import Manager
from .offline import restrict_ui_network
from .registry import availability, load_registry
from .runtime import Runtime

CODING_PATHS = {
    "/v1/chat/completions", "/v1/responses", "/v1/responses/input_tokens",
    "/v1/messages", "/v1/messages/count_tokens",
}


async def read_body(request, limit):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(413, "Request too large")
    return bytes(body)


async def finish_thread(function, *args):
    """Do not release GPU ownership while a cancelled thread still uses it."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise


def create_app(manager):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    gpu = asyncio.Lock()
    waiting = 0
    started = None

    @asynccontextmanager
    async def slot():
        nonlocal waiting, started
        if waiting >= int(os.environ["UI_QUEUE"]):
            raise HTTPException(429, "GPU queue is full")
        waiting += 1
        try:
            async with gpu:
                started = time.monotonic()
                try:
                    yield
                finally:
                    started = None
        finally:
            waiting -= 1

    @app.get("/health")
    async def health():
        process = getattr(manager, "process", None)
        profile = manager.registry.get(getattr(manager, "active", None), {}) if process and process.poll() is None else {}
        progress = None
        progress_path = getattr(manager, "image_progress_path", None)
        if gpu.locked() and progress_path is not None:
            try:
                progress = json.loads(progress_path.read_text())
            except (OSError, ValueError):
                pass
        return {"status": "ok", "epoch": manager.runtime.name,
                "busy": gpu.locked(), "queued": max(0, waiting - int(gpu.locked())),
                "progress": progress,
                "engine": profile.get("engine"), "active_model": profile.get("label", "").split(" · ")[0] or None,
                "elapsed_seconds": int(time.monotonic() - started) if started is not None else 0}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": os.environ["MODEL_ID"], "object": "model", "owned_by": "local"}]}

    @app.post("/v1/{path:path}")
    async def coding(path: str, request: Request):
        target = "/v1/" + path
        if target not in CODING_PATHS:
            raise HTTPException(404, "Unsupported coding route")
        body = await read_body(request, int(os.environ["STACK_MAX_BODY"]))
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict) or payload.get("model") != os.environ["MODEL_ID"]:
                raise ValueError()
            # Model switching discards engine-owned response state. Require explicit input.
            if payload.get("previous_response_id") or payload.get("background") or payload.get("store"):
                raise ValueError()
            if target == "/v1/responses" and "store" not in payload:
                payload["store"] = False
                body = json.dumps(payload, ensure_ascii=False).encode()
        except (ValueError, TypeError):
            raise HTTPException(400, "Use the advertised model and stateless requests") from None
        ownership = slot()
        await ownership.__aenter__()
        upstream = None
        client = httpx.AsyncClient(timeout=float(os.environ["UI_REQUEST_TIMEOUT"]), trust_env=False)
        try:
            await finish_thread(manager.ensure_chat, os.environ["CODING_PROFILE"], int(os.environ["UI_CONTEXT"]))
            headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
            for key in ("anthropic-version", "anthropic-beta"):
                if key in request.headers:
                    headers[key] = request.headers[key]
            upstream = await client.send(client.build_request("POST", manager.url + target, content=body, headers=headers), stream=True)
        except BaseException as exc:
            with anyio.CancelScope(shield=True):
                await client.aclose()
                await ownership.__aexit__(None, None, None)
            if isinstance(exc, (httpx.HTTPError, RuntimeError, TimeoutError, ValueError, KeyError)):
                raise HTTPException(503, "Coding inference is unavailable. Check the host inference service.") from None
            raise

        async def stream():
            completed = False
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
                completed = True
            finally:
                with anyio.CancelScope(shield=True):
                    try:
                        await upstream.aclose()
                        await client.aclose()
                        # Retire an interrupted engine before another GPU request.
                        if not completed:
                            await finish_thread(manager.stop)
                    finally:
                        await ownership.__aexit__(None, None, None)

        return StreamingResponse(stream(), status_code=upstream.status_code,
                                 headers={"Content-Type": upstream.headers.get("content-type", "application/json"),
                                          "Cache-Control": "no-store"})

    @app.get("/images/options")
    async def options():
        profiles = [{"id": key, "label": p["label"], "sizes": images.size_choices(p)}
                    for key, p in manager.registry.items() if p["kind"] == "image" and availability(p) == "Installed"]
        return {"models": profiles, "steps": int(os.environ["IMAGE_STEPS"]),
                "max_steps": int(os.environ["IMAGE_UI_MAX_STEPS"]),
                "max_references": int(os.environ["IMAGE_MAX_REFERENCES"]), "epoch": manager.runtime.name}

    @app.post("/images/generate")
    async def generate(request: Request):
        raw = await read_body(request, int(os.environ["STACK_MAX_BODY"]))
        try:
            body = json.loads(raw)
            key = body["model"]
            if key not in manager.registry or manager.registry[key]["kind"] != "image":
                raise ValueError()
            references = body.get("references", [])
            if not isinstance(references, list) or len(references) > int(os.environ["IMAGE_MAX_REFERENCES"]):
                raise ValueError()
            if not isinstance(body["prompt"], str):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise HTTPException(400, "Invalid image request") from None
        async with slot():
            def run():
                upload = Path(tempfile.mkdtemp(prefix="upload-", dir=manager.runtime))
                result_dir = None
                try:
                    paths = []
                    for index, encoded in enumerate(references):
                        path = upload / f"{index}.png"
                        path.write_bytes(base64.b64decode(encoded, validate=True))
                        paths.append(str(path))
                    result, status, _ = images.generate(manager, key, body["prompt"], body["size"],
                                                        body["steps"], body["seed"], paths)
                    result_dir = Path(result).parent
                    return {"image": base64.b64encode(Path(result).read_bytes()).decode(), "status": status,
                            "epoch": manager.runtime.name}
                finally:
                    shutil.rmtree(upload)
                    if result_dir:
                        shutil.rmtree(result_dir)
                        result_dir.with_suffix(".log").unlink(missing_ok=True)
            try:
                return await finish_thread(run)
            except (ValueError, KeyError, TypeError, OverflowError):
                raise HTTPException(400, "Invalid image settings or reference") from None
            except (RuntimeError, TimeoutError):
                raise HTTPException(503, "Image inference failed. Check the host inference service.") from None

    return app


def main():
    import uvicorn

    socket = Path(os.environ["INFERENCE_SOCKET"])
    socket.parent.mkdir(parents=True, exist_ok=True)
    owner = (socket.parent / ".owner.lock").open("a")
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        owner.close()
        try:
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(socket)),
                              base_url="http://inference", timeout=2, trust_env=False) as client:
                health = client.get("/health")
                health.raise_for_status()
                if health.json().get("status") != "ok":
                    raise ValueError("Unexpected health response")
        except (httpx.HTTPError, ValueError):
            raise SystemExit("Inference is already starting or is unresponsive. Check the existing service; no second worker was started.") from None
        print("cook-4090: inference is already running and healthy. No second worker was started.")
        print("To change its configuration, stop the existing service first.")
        return
    # Refuse to remove a different program's live socket or a non-socket file.
    if socket.exists() or socket.is_symlink():
        if socket.is_symlink() or not stat.S_ISSOCK(socket.stat().st_mode):
            raise RuntimeError("Inference socket path contains another file")
        with sockets.socket(sockets.AF_UNIX) as probe:
            probe.settimeout(1)
            try:
                probe.connect(str(socket))
            except ConnectionRefusedError:
                socket.unlink()
            else:
                raise RuntimeError("Inference socket is already in use")
    runtime = Runtime(os.environ["UI_RUNTIME"])
    manager = Manager(load_registry(os.environ["UI_REGISTRY"]), runtime.path, live_logs=True)
    restrict_ui_network()

    def terminate(*_):
        # Uvicorn replays shutdown signals after restoring previous handlers.
        # Convert them into Python exit so ownership and artifacts are cleaned.
        raise SystemExit(0)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, terminate)
    listener = sockets.socket(sockets.AF_UNIX)
    try:
        old_umask = os.umask(0o077)
        try:
            listener.bind(str(socket))
            socket.chmod(0o600)
        finally:
            os.umask(old_umask)
        connections = int(os.environ["INFERENCE_CONNECTIONS"])
        listener.listen(connections)
        print("cook-4090: foreground inference · " + manager.registry[os.environ["CODING_PROFILE"]]["label"], flush=True)
        print("Models load on request. Engine logs appear here. Ctrl+C stops inference and releases the GPU.", flush=True)
        server = uvicorn.Server(uvicorn.Config(create_app(manager), access_log=False, limit_concurrency=connections))
        server.run(sockets=[listener])
    finally:
        listener.close()
        manager.close()
        socket.unlink(missing_ok=True)
        runtime.close()
        owner.close()


if __name__ == "__main__":
    main()

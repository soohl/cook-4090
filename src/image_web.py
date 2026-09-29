"""Offline image workspace: React assets, saved sessions, and host inference over UDS."""

import asyncio
import base64
import json
import os
from pathlib import Path
import tempfile
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from . import ROOT
from .image_history import ImageHistory
from .image_edit import annotate_edit, composite_edit, png, prepare_edit
from .offline import environment

os.environ.update(environment())
_runtime = tempfile.TemporaryDirectory(prefix="cook-4090-web-")
RUNTIME = Path(_runtime.name)
HISTORY = Path(os.environ.get("IMAGE_HISTORY_DIR", str(RUNTIME / "history")))
FRONTEND = ROOT / "web/dist"
store = ImageHistory(HISTORY)
EPOCH = uuid.uuid4().hex
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
jobs = {}
tasks = set()
gpu_queue = asyncio.Semaphore(1)


@app.middleware("http")
async def headers(request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("sec-fetch-site") not in {None, "same-origin"}:
        return Response(status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; connect-src 'self'; img-src 'self' data: blob:; "
        "font-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; "
        "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    )
    return response


async def upstream(method, target, body=b""):
    transport = httpx.AsyncHTTPTransport(uds=os.environ["INFERENCE_SOCKET"])
    async with httpx.AsyncClient(transport=transport, timeout=3 if method == "GET" else float(os.environ["IMAGE_WEB_TIMEOUT"]), trust_env=False) as client:
        return await client.request(method, "http://inference" + target, content=body,
                                    headers={"Content-Type": "application/json"})


def with_epoch(payload):
    if "epoch" in payload:
        payload["epoch"] += ":" + EPOCH
    return payload


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/api/options")
@app.get("/api/health")
async def inference(request: Request):
    operation = request.url.path.rsplit("/", 1)[-1]
    target = {"options": "/images/options", "health": "/health"}.get(operation)
    if target is None:
        raise HTTPException(404)
    try:
        result = await upstream("GET", target)
        return Response(json.dumps(with_epoch(result.json())), result.status_code, media_type="application/json")
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, "Inference is offline. Start ./run.sh on the server.") from None


def session_entries(key):
    try:
        entries = store.read(key)
    except ValueError:
        raise HTTPException(404) from None
    if not entries and key not in jobs:
        raise HTTPException(404)
    return entries


def public_session(key):
    entries = session_entries(key)
    return {"id": key, "status": jobs.get(key, {}).get("status", "idle"),
            "error": jobs.get(key, {}).get("error"),
            "entries": [dict({k: e[k] for k in ("prompt", "model", "size", "steps", "seed", "status")},
                             image=f"/api/sessions/{key}/images/{i}",
                             edit={"base": e["edit"]["base"], "feather": e["edit"].get("feather", 12), "mask": f"/api/sessions/{key}/masks/{i}"} if e.get("edit") else None,
                             references=[f"/api/sessions/{key}/references/{i}/{j}" for j in range(len(e["references"]))])
                        for i, e in enumerate(entries)]}


@app.get("/api/sessions")
async def list_sessions():
    saved = dict((key, title) for title, key in store.choices())
    keys = list(dict.fromkeys([*reversed(jobs), *saved]))
    return [{"id": key, "title": saved.get(key, jobs.get(key, {}).get("prompt", "New session"))[:60],
             "status": jobs.get(key, {}).get("status", "idle")} for key in keys]


# A dedicated route avoids exposing filesystem paths in session metadata.
@app.get("/api/sessions/{key}")
async def get_session(key: str):
    return public_session(key)


@app.delete("/api/sessions/{key}", status_code=204)
async def delete_session(key: str):
    session_entries(key)
    if jobs.get(key, {}).get("status") in {"queued", "generating"}:
        raise HTTPException(409, "Wait for this generation to finish before deleting the session.")
    store.delete(key)
    jobs.pop(key, None)
    return Response(status_code=204)


def saved_file(key, index, reference=None, mask=False):
    entries = session_entries(key)
    try:
        if index < 0 or (reference is not None and reference < 0):
            raise IndexError()
        value = entries[index]["edit"]["mask"] if mask else entries[index]["image"] if reference is None else entries[index]["references"][reference]
        path = Path(value).resolve()
        if not path.is_relative_to(store.directory(key).resolve()) or not path.is_file():
            raise IndexError()
    except (IndexError, KeyError, TypeError, ValueError):
        raise HTTPException(404) from None
    return FileResponse(path, filename=path.name, content_disposition_type="inline")


@app.get("/api/sessions/{key}/images/{index}")
async def image_file(key: str, index: int):
    return saved_file(key, index)


@app.get("/api/sessions/{key}/references/{index}/{reference}")
async def reference_file(key: str, index: int, reference: int):
    return saved_file(key, index, reference)


@app.get("/api/sessions/{key}/masks/{index}")
async def mask_file(key: str, index: int):
    return saved_file(key, index, mask=True)


class AreaEdit(BaseModel):
    base: int = Field(ge=0, le=9, strict=True)
    mask: str = Field(min_length=1)
    feather: int = Field(default=12, ge=0, le=40, strict=True)


class Generation(BaseModel):
    session: str | None = None
    model: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=1, max_length=4000)
    size: str = Field(pattern=r"^\d{2,5}x\d{2,5}$")
    steps: int = Field(ge=1, le=1000)
    seed: int = Field(ge=-1, le=4294967295)
    references: list[str] = Field(default_factory=list, max_length=10)
    edit: AreaEdit | None = None


async def generate(key, body, prepared=None):
    try:
        async with gpu_queue:
            jobs[key]["status"] = "generating"
            with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
                references = []
                for index, value in enumerate(body.references):
                    path = Path(directory) / f"reference-{index}.png"
                    path.write_bytes(base64.b64decode(value, validate=True))
                    references.append(str(path))
                payload = body.model_dump(exclude={"session", "edit"})
                mask_bytes = None
                if prepared:
                    base, mask, instruction = prepared
                    mask_bytes = png(mask)
                    payload["references"] = list(body.references)
                    payload["references"][body.edit.base] = base64.b64encode(annotate_edit(base, mask)).decode()
                    payload["prompt"] = instruction
                if payload["seed"] == -1:
                    import secrets
                    payload["seed"] = secrets.randbelow(2**32)
                result = await upstream("POST", "/images/generate", json.dumps(payload).encode())
                if not result.is_success:
                    raise RuntimeError(result.json().get("detail", "Image generation failed."))
                result = result.json()
                entry = {k: payload[k] for k in ("model", "prompt", "size", "steps", "seed")}
                entry["prompt"] = body.prompt
                entry["status"] = result["status"]
                output = base64.b64decode(result["image"], validate=True)
                if prepared:
                    output = await asyncio.to_thread(composite_edit, output, base, mask, tuple(map(int, body.size.split("x"))), body.edit.feather)
                    entry["edit"] = {"base": body.edit.base, "feather": body.edit.feather}
                store.append(key, entry, output, references, mask=mask_bytes)
                jobs[key] = dict(status="idle", prompt=body.prompt)
    except asyncio.CancelledError:
        jobs[key].update(status="error", error="Generation was interrupted. Please try again.")
        raise
    except (httpx.HTTPError, OSError, ValueError, RuntimeError, KeyError):
        jobs[key].update(status="error", error="Generation failed. Check that inference is running, then try again.")


@app.post("/api/generate", status_code=202)
async def start_generation(request: Request):
    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        raise HTTPException(415)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > int(os.environ["STACK_MAX_BODY"]):
            raise HTTPException(413)
    try:
        body = Generation.model_validate_json(raw)
        if not body.prompt.strip():
            raise ValueError()
        key = body.session or uuid.uuid4().hex
        store.directory(key)
        for ref in body.references:
            base64.b64decode(ref, validate=True)
    except (ValidationError, ValueError):
        raise HTTPException(400, "Check your prompt, settings, and reference images.") from None
    if jobs.get(key, {}).get("status") in {"queued", "generating"}:
        raise HTTPException(409, "This session already has a generation in progress.")
    if sum(job["status"] in {"queued", "generating"} for job in jobs.values()) >= int(os.environ["UI_QUEUE"]):
        raise HTTPException(429, "The image queue is full. Please try again shortly.")
    prepared = None
    if body.edit:
        try:
            prepared = await asyncio.to_thread(prepare_edit, body.references, body.edit.base, body.edit.mask, body.prompt)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        # Validation yields to other requests. Recheck before claiming a queue slot.
        if jobs.get(key, {}).get("status") in {"queued", "generating"}:
            raise HTTPException(409, "This session already has a generation in progress.")
        if sum(job["status"] in {"queued", "generating"} for job in jobs.values()) >= int(os.environ["UI_QUEUE"]):
            raise HTTPException(429, "The image queue is full. Please try again shortly.")
    jobs[key] = dict(status="queued", prompt=body.prompt)
    task = asyncio.create_task(generate(key, body, prepared))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {"id": key}


@app.get("/")
async def index():
    if not (FRONTEND / "index.html").is_file():
        raise HTTPException(503, "The web interface has not been built.")
    return FileResponse(FRONTEND / "index.html")


app.mount("/assets", StaticFiles(directory=FRONTEND / "assets", check_dir=False), name="assets")


def main():
    import uvicorn
    from .offline import restrict_ui_network
    restrict_ui_network()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["IMAGE_WEB_PORT"]),
                access_log=False, proxy_headers=False, limit_concurrency=int(os.environ["IMAGE_WEB_CONNECTIONS"]))


if __name__ == "__main__":
    main()

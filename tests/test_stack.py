"""CPU checks for the private coding boundary and Docker image application."""

import asyncio
import base64
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from src import image_web
from src.inference_service import create_app, finish_thread
from src.stack import initialize, load_env, validate_bind, refresh_docker_group


class ConfigurationTests(unittest.TestCase):
    def test_refreshes_only_existing_docker_membership(self):
        from types import SimpleNamespace
        import shlex
        with patch.dict(os.environ, {}, clear=True), patch('src.stack.Path.exists', return_value=True), \
             patch('src.stack.os.access', return_value=False), \
             patch('src.stack.Path.stat', return_value=SimpleNamespace(st_gid=999)), \
             patch('grp.getgrgid', return_value=SimpleNamespace(gr_name='docker', gr_gid=999)), \
             patch('pwd.getpwuid', return_value=SimpleNamespace(pw_name='fixture', pw_gid=1000)), \
             patch('src.stack.os.getgid', return_value=1000), patch('src.stack.os.getgroups', return_value=[1000]), \
             patch('src.stack.os.getgrouplist', return_value=[1000, 999]) as memberships, \
             patch('src.stack.shutil.which', return_value='/usr/bin/sg'), \
             patch('src.stack.os.execvp') as execute, patch('src.stack.sys.argv', ['commands', '--verify']):
            refresh_docker_group('src.commands')
            self.assertEqual(execute.call_args.args[1][:3], ['sg', 'docker', '-c'])
            self.assertEqual(shlex.split(execute.call_args.args[1][3])[1:], ['-m', 'src.commands', '--verify'])
            execute.reset_mock()
            memberships.return_value = [1000]
            refresh_docker_group('src.commands')
            execute.assert_not_called()
            memberships.return_value = [1000, 999]
            with patch.dict(os.environ, DOCKER_HOST='unix:///fixture.sock'):
                refresh_docker_group('src.commands')
            execute.assert_not_called()

    def test_reject_public_and_wildcard_bind(self):
        for value in ("0.0.0.0", "::", "::1", "8.8.8.8", "169.254.1.1", "100.64.1.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_bind(value)
        for value in ("127.0.0.1", "10.0.0.8", "172.16.2.3", "192.168.1.10"):
            self.assertEqual(validate_bind(value), value)

    def test_credentials_are_private_unique_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            initialize(path)
            first = load_env(path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotEqual(first["CODING_API_KEY"], first["POCKET_ID_ENCRYPTION_KEY"])
            self.assertEqual(len(base64.urlsafe_b64decode(first["IMAGE_COOKIE_SECRET"])), 32)
            with self.assertRaises(FileExistsError):
                initialize(path)
            self.assertEqual(first, load_env(path))
            path.write_text('CODING_API_KEY=$(touch injected)\n')
            with self.assertRaises(ValueError):
                load_env(path)


class FakeManager:
    registry = {"image": {"kind": "image", "engine": "diffusers", "label": "Image", "env": {}, "required_env": []}}
    url = "http://127.0.0.1:12345"

    def __init__(self, directory):
        self.runtime = Path(directory)
        self.lock = threading.Lock()
        self.loads = []
        self.stopped = False

    def ensure_chat(self, *args):
        self.loads.append(args)

    def stop(self):
        self.stopped = True


class InferenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.manager = FakeManager(self.directory.name)
        self.app = create_app(self.manager)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.real_client = httpx.AsyncClient

    async def asyncTearDown(self):
        await self.client.aclose()
        self.directory.cleanup()

    def upstream(self, handler):
        return patch("src.inference_service.httpx.AsyncClient",
                     side_effect=lambda **kwargs: self.real_client(transport=httpx.MockTransport(handler), **kwargs))

    async def test_forwards_tool_schema_raw_sse_and_anthropic_headers(self):
        raw = b'{"model":"qwen3.8-27b","messages":[],"tools":[{"z":1,"a":2}],"stream":true}'
        events = b'data: {"choices":[{"delta":{"tool_calls":[{"id":"a","function":{"arguments":"{}"}}]}}]}\n\ndata: [DONE]\n\n'
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield events[:20]
                yield events[20:]
        def handler(request):
            self.assertEqual(request.content, raw)
            self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
            self.assertNotIn("authorization", request.headers)
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=Stream())
        with self.upstream(handler):
            result = await self.client.post("/v1/chat/completions", content=raw,
                                            headers={"anthropic-version": "2023-06-01", "Authorization": "secret"})
        self.assertEqual(result.content, events)
        self.assertEqual(result.headers["content-type"], "text/event-stream")
        self.assertEqual(len(self.manager.loads), 1)

    async def test_unknown_routes_and_stateful_requests_do_not_start_engine(self):
        for path, payload, status in [
            ("/v1/arbitrary", {}, 404),
            ("/v1/responses", {"model": "wrong"}, 400),
            ("/v1/responses", {"model": "qwen3.8-27b", "previous_response_id": "r"}, 400),
            ("/v1/responses", {"model": "qwen3.8-27b", "store": True}, 400),
        ]:
            result = await self.client.post(path, json=payload)
            self.assertEqual(result.status_code, status)
        self.assertFalse(self.manager.loads)

    async def test_responses_disable_storage_and_return_engine_error(self):
        def handler(request):
            self.assertIs(json.loads(request.content)["store"], False)
            return httpx.Response(422, stream=BytesStream(b'{"error":"bad tool"}'))
        with self.upstream(handler):
            result = await self.client.post("/v1/responses", json={"model": "qwen3.8-27b", "input": "hello"})
        self.assertEqual(result.status_code, 422)
        self.assertEqual(result.json(), {"error": "bad tool"})

    async def test_failure_releases_gpu_and_body_limit_precedes_load(self):
        with patch.object(self.manager, "ensure_chat", side_effect=RuntimeError("private engine log")):
            result = await self.client.post("/v1/chat/completions", json={"model": "qwen3.8-27b"})
            self.assertEqual(result.status_code, 503)
            self.assertNotIn("private engine log", result.text)
        with patch.dict(os.environ, STACK_MAX_BODY="5"):
            result = await self.client.post("/v1/chat/completions", content=b"123456")
            self.assertEqual(result.status_code, 413)
        with self.upstream(lambda request: httpx.Response(200, stream=BytesStream(b'{}'))):
            result = await asyncio.wait_for(self.client.post("/v1/chat/completions", json={"model": "qwen3.8-27b"}), 3)
            self.assertEqual(result.status_code, 200)

    async def test_queue_holds_gpu_until_stream_ends(self):
        entered, finish = asyncio.Event(), asyncio.Event()
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await finish.wait()
                yield b'data: [DONE]\n\n'
        with patch.dict(os.environ, UI_QUEUE="1"), self.upstream(lambda request: httpx.Response(200, stream=SlowStream())):
            first = asyncio.create_task(self.client.post("/v1/chat/completions", json={"model": "qwen3.8-27b"}))
            await asyncio.wait_for(entered.wait(), 3)
            try:
                health = (await self.client.get("/health")).json()
                self.assertTrue(health["busy"])
                self.assertEqual(health["queued"], 0)
                self.assertGreaterEqual(health["elapsed_seconds"], 0)
                self.assertIsNone(health["progress"])
                self.manager.image_progress_path = self.manager.runtime / "progress.json"
                progress = {"completed": 12, "total": 25}
                self.manager.image_progress_path.write_text(json.dumps(progress))
                self.assertEqual((await self.client.get("/health")).json()["progress"], progress)
                second = await self.client.post("/images/generate", json={"model": "image", "prompt": "hello"})
                self.assertEqual(second.status_code, 429)
                self.assertEqual(len(self.manager.loads), 1)
            finally:
                finish.set()
                await first
            self.assertFalse((await self.client.get("/health")).json()["busy"])
            self.assertIsNone((await self.client.get("/health")).json()["progress"])

    async def test_disconnected_stream_retires_engine_and_releases_gpu(self):
        entered = asyncio.Event()
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                yield b"data: partial\n\n"
                await asyncio.Event().wait()
        with self.upstream(lambda request: httpx.Response(200, stream=SlowStream())):
            request = asyncio.create_task(self.client.post("/v1/chat/completions", json={"model": "qwen3.8-27b"}))
            await asyncio.wait_for(entered.wait(), 3)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
        self.assertTrue(self.manager.stopped)
        with self.upstream(lambda request: httpx.Response(200, stream=BytesStream(b'{}'))):
            result = await asyncio.wait_for(self.client.post("/v1/chat/completions", json={"model": "qwen3.8-27b"}), 3)
        self.assertEqual(result.status_code, 200)

    async def test_image_result_is_returned_and_files_are_removed(self):
        def generate(manager, key, prompt, size, steps, seed, references):
            self.assertEqual(Path(references[0]).read_bytes(), b"fake-reference")
            directory = manager.runtime / "image-result"
            directory.mkdir()
            (directory / "image.png").write_bytes(b"fake-png")
            directory.with_suffix(".log").write_text("private")
            return str(directory / "image.png"), "ok", str(directory / "report.json")
        with patch("src.inference_service.images.generate", side_effect=generate):
            result = await self.client.post("/images/generate", json={"model": "image", "prompt": "hello", "size": "1024x1024",
                "steps": 1, "seed": 42, "references": [base64.b64encode(b"fake-reference").decode()]})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(base64.b64decode(result.json()["image"]), b"fake-png")
        self.assertEqual(list(self.manager.runtime.iterdir()), [])

    async def test_cancelled_thread_finishes_before_gpu_can_be_released(self):
        started, release = threading.Event(), threading.Event()
        def work():
            started.set()
            release.wait(3)
        task = asyncio.create_task(finish_thread(work))
        await asyncio.to_thread(started.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content

    async def __aiter__(self):
        yield self.content


class ImageWebTests(unittest.TestCase):
    def test_public_app_has_no_coding_or_host_file_routes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(image_web, 'FRONTEND', Path(directory)), TestClient(image_web.app) as client:
            (Path(directory) / 'index.html').write_text('<div id="root"></div>')
            for path in ("/v1/models", "/api/chat/completions", "/api/arbitrary", "/app.py", "/docs", "/openapi.json"):
                self.assertEqual(client.get(path).status_code, 404)
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertEqual(page.headers["cache-control"], "no-store")
            self.assertIn("frame-ancestors 'none'", page.headers["content-security-policy"])
            self.assertIn('id="root"', page.text)
            self.assertNotIn("unsafe-eval", page.headers["content-security-policy"])
            self.assertEqual(client.get("/gradio_api/file=/etc/passwd").status_code, 404)
            self.assertEqual(client.post("/api/generate", data={"prompt": "hi"}).status_code, 415)
            self.assertEqual(client.post("/api/generate", json={}, headers={"sec-fetch-site": "cross-site"}).status_code, 403)

    def test_web_restart_changes_browser_epoch(self):
        real_client = httpx.AsyncClient
        with TestClient(image_web.app) as client:
            def upstream(**kwargs):
                kwargs.pop("transport")
                return real_client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"epoch": "host"})), **kwargs)
            with patch("src.image_web.httpx.AsyncClient", side_effect=upstream), patch.object(image_web, "EPOCH", "web-one"):
                first = client.get("/api/health").json()["epoch"]
            with patch("src.image_web.httpx.AsyncClient", side_effect=upstream), patch.object(image_web, "EPOCH", "web-two"):
                second = client.get("/api/health").json()["epoch"]
        self.assertEqual(first, "host:web-one")
        self.assertEqual(second, "host:web-two")

class HostLifecycleTests(unittest.TestCase):
    def test_host_uses_private_socket_and_cleans_it_on_shutdown(self):
        import subprocess
        import sys
        import time
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            socket = root / "ipc/inference.sock"
            runtime = root / "runtime"
            env = dict(os.environ, INFERENCE_SOCKET=str(socket), UI_RUNTIME=str(runtime))
            with (root / "service.log").open("w+") as log:
                process = subprocess.Popen([sys.executable, "-m", "src.inference_service"], env=env, stdout=log, stderr=log)
                try:
                    with httpx.Client(transport=httpx.HTTPTransport(uds=str(socket)), base_url="http://inference", timeout=1) as client:
                        deadline = time.monotonic() + 10
                        while True:
                            try:
                                response = client.get("/health")
                                break
                            except httpx.TransportError:
                                if process.poll() is not None or time.monotonic() > deadline:
                                    log.seek(0)
                                    self.fail(log.read())
                                time.sleep(.05)
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(socket.stat().st_mode & 0o777, 0o600)
                        self.assertEqual(client.get("/v1/models").json()["data"][0]["id"], os.environ["MODEL_ID"])
                        self.assertEqual(len(list(runtime.glob("session-*"))), 1)
                        duplicate = subprocess.run([sys.executable, "-m", "src.inference_service"],
                            env=env, text=True, capture_output=True, timeout=10)
                        self.assertEqual(duplicate.returncode, 0, duplicate.stderr)
                        self.assertIn("already running and healthy", duplicate.stdout)
                        self.assertEqual(client.get("/health").json()["epoch"], response.json()["epoch"])
                        self.assertIsNone(process.poll())
                finally:
                    process.terminate()
                    process.wait(timeout=15)
                self.assertFalse(socket.exists())
                self.assertFalse(list(runtime.glob("session-*")))

@unittest.skipUnless(os.environ.get("CADDY_TEST_BINARY"), "Set CADDY_TEST_BINARY for native edge checks")
class CaddyBoundaryTests(unittest.TestCase):
    def test_real_edge_authenticates_and_blocks_image_and_admin_paths(self):
        from http.server import BaseHTTPRequestHandler
        import socket
        import socketserver
        import subprocess
        import time
        from src import ROOT

        key = "a" * 32
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append((self.path, dict(self.headers)))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unix = str(root / "inference.sock")
            server = socketserver.UnixStreamServer(unix, Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            caddyfile = (ROOT / "config/Caddyfile").read_text()
            for name in ("image-auth.caddy", "identity.caddy"):
                caddyfile = caddyfile.replace("/etc/caddy/" + name, str(ROOT / "config" / name))
            caddyfile = caddyfile.replace("/inference/inference.sock", unix)
            (root / "Caddyfile").write_text(caddyfile)
            binary = os.environ["CADDY_TEST_BINARY"]
            env = dict(os.environ, FUNNEL_HOSTNAME="test.example.ts.net", CODING_API_KEY=key)
            config = json.loads(subprocess.check_output([binary, "adapt", "--config", str(root / "Caddyfile")], env=env))
            coding = next(s for s in config["apps"]["http"]["servers"].values() if ":8000" in s["listen"])
            with socket.socket() as allocation:
                allocation.bind(("127.0.0.1", 0))
                port = allocation.getsockname()[1]
            coding["listen"] = [f"127.0.0.1:{port}"]
            config["apps"]["http"]["servers"] = {"coding": coding}
            config["apps"].pop("tls", None)
            (root / "caddy.json").write_text(json.dumps(config))
            with (root / "edge.log").open("w+") as log:
                process = subprocess.Popen([binary, "run", "--config", str(root / "caddy.json")], env=env, stdout=log, stderr=log)
                try:
                    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=1) as client:
                        deadline = time.monotonic() + 10
                        while True:
                            try:
                                result = client.get("/v1/models")
                                break
                            except httpx.TransportError:
                                if process.poll() is not None or time.monotonic() > deadline:
                                    log.seek(0)
                                    self.fail(log.read())
                                time.sleep(.05)
                        self.assertEqual(result.status_code, 401)
                        self.assertEqual(client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code, 401)
                        for auth in ({"Authorization": "Bearer " + key}, {"x-api-key": key}):
                            self.assertEqual(client.get("/v1/models", headers=auth).status_code, 200)
                            for path in ("/images/options", "/images/generate", "/health", "/admin", "/v1/models/../../images/options"):
                                self.assertEqual(client.get(path, headers=auth).status_code, 404)
                        self.assertEqual(len(seen), 2)
                        self.assertTrue(all("Authorization" not in headers and "X-Api-Key" not in headers for _, headers in seen))
                finally:
                    process.terminate()
                    process.wait(timeout=10)
                    server.shutdown()
                    server.server_close()
                    thread.join()

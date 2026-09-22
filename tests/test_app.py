"""CPU-only checks of ownership, privacy, offline controls, and matched metrics."""

import io
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from src import benchmarks, client, images
from src.manager import Manager
from src.registry import ROOT, load_registry
from src.runtime import Runtime


class ImageTests(unittest.TestCase):
    def test_image_profiles_support_both_engines_and_reject_chat_mismatch(self):
        registry = load_registry(ROOT / "config/models.json")
        self.assertEqual({p["engine"] for p in registry.values() if p["kind"] == "image"}, {"diffusers", "sglang"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps([{"id": "bad", "kind": "chat", "engine": "sglang"}]))
            with self.assertRaisesRegex(ValueError, "Unsupported engine"):
                load_registry(path)

    def test_native_presets_and_model_specific_resolution_validation(self):
        native = {"2048x2048", "2400x1792", "1792x2400", "2528x1696",
                  "1696x2528", "2752x1536", "1536x2752"}
        self.assertTrue(native <= set(images.size_choices({})))
        profile = {"kind": "image", "env": {"IMAGE_UI_SIZES": "1536x2752"}}

        class FakeManager:
            registry = {"portrait": profile}
            lock = threading.Lock()

            def run_image(self, key, env, log):
                self.env = env
                output = Path(env["IMAGE_OUTPUT"])
                output.mkdir()
                (output / "report.json").write_text('{"generation_seconds": 1.0}')

        with tempfile.TemporaryDirectory() as directory:
            manager = FakeManager()
            manager.runtime = Path(directory)
            with self.assertRaisesRegex(ValueError, "listed image size"):
                images.generate(manager, "portrait", "A mountain", "1024x1024", 40, 42, [])
            images.generate(manager, "portrait", "A mountain", "1536x2752", 40, 42, [])
            self.assertEqual((manager.env["IMAGE_WIDTH"], manager.env["IMAGE_HEIGHT"]), ("1536", "2752"))
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            images.size_choices({"env": {"IMAGE_UI_SIZES": "2064x2048"}})


class RuntimeTests(unittest.TestCase):
    def test_cleans_stale_sessions_only_and_enforces_single_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session-stale").mkdir()
            (root / "session-stale" / "chat").write_text("private")
            (root / "keep").write_text("unrelated")
            runtime = Runtime(root)
            self.assertFalse((root / "session-stale").exists())
            with self.assertRaises(RuntimeError):
                Runtime(root)
            artifact = runtime.path / "image.png"
            artifact.write_text("private")
            runtime.close()
            self.assertFalse(artifact.exists())
            self.assertTrue((root / "keep").exists())
            Runtime(root).close()


class OfflineTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "Native offline guard requires a C compiler")
    def test_native_local_ipc_allows_local_workers_and_rejects_internet(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Path(directory) / "connect.so"
            subprocess.run(["cc", "-shared", "-fPIC", str(ROOT / "src/offline_connect.c"), "-ldl", "-o", str(guard)], check=True)
            code = """
import socket, subprocess, sys
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0)); listener.listen()
    with socket.create_connection(listener.getsockname()): pass
with socket.socket(socket.AF_UNIX) as listener:
    listener.bind('\\0cook-4090-test-' + str(__import__('os').getpid())); listener.listen()
    with socket.socket(socket.AF_UNIX) as client: client.connect(listener.getsockname())
for address in [('192.0.2.1', 443), ('2001:db8::1', 443), ('::ffff:192.0.2.1', 443)]:
    family = socket.AF_INET6 if ':' in address[0] else socket.AF_INET
    with socket.socket(family) as client:
        try: client.connect(address)
        except PermissionError: pass
        else: raise AssertionError('Internet connect allowed')
try: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except PermissionError: pass
else: raise AssertionError('Internet datagram allowed')
child = subprocess.run([sys.executable, '-c', "import socket; socket.socket().connect(('192.0.2.1', 443))"], capture_output=True)
assert child.returncode != 0 and b'Operation not permitted' in child.stderr
print('pass')
"""
            result = subprocess.run([sys.executable, "-m", "src.offline", "--local-ipc", sys.executable, "-c", code],
                                    cwd=ROOT, env=dict(os.environ, SGLANG_OFFLINE_LIB=str(guard)),
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def execute(self, code, worker=False):
        command = [sys.executable, "-c", code]
        if worker:
            command = [sys.executable, "-m", "src.offline", *command]
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)

    def test_worker_cannot_connect_or_create_internet_datagram_socket(self):
        code = """
import socket
for fn in [lambda: socket.socket().connect(('127.0.0.1', 9)),
           lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
           lambda: socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)]:
    try: fn()
    except PermissionError: pass
    else: raise AssertionError('outbound connection allowed')
with socket.socket() as s:
    s.bind(('127.0.0.1', 0)); s.listen()
print('pass')
"""
        result = self.execute(code, worker=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ui_rejects_nonlocal_connections_and_dns_before_network(self):
        code = """
import socket
from src.offline import restrict_ui_network
restrict_ui_network()
for fn in [lambda: socket.socket().connect(('192.0.2.1', 443)),
           lambda: socket.getaddrinfo('example.com', 443),
           lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'test', ('192.0.2.1', 53))]:
    try: fn()
    except PermissionError: pass
    else: raise AssertionError('outbound connection allowed')
print('pass')
"""
        result = self.execute(code)
        self.assertEqual(result.returncode, 0, result.stderr)


class StreamTests(unittest.TestCase):
    def test_reasoning_usage_and_incomplete_stream(self):
        events = [
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]},
            {"usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        ]
        data = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data + b"data: [DONE]\n")):
            result = list(client.stream("http://127.0.0.1", {}, 1))[-1]
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["reasoning"], "think")
        self.assertEqual(result["usage"]["completion_tokens"], 3)
        self.assertTrue(result["done"])
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data)):
            with self.assertRaises(RuntimeError):
                list(client.stream("http://127.0.0.1", {}, 1))


class ManagerTests(unittest.TestCase):
    def test_image_engines_route_to_their_environment_and_release_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles = {engine: {"kind": "image", "engine": engine, "label": engine, "required_env": []}
                        for engine in ("diffusers", "sglang")}
            manager = Manager(profiles, directory)
            env = dict(os.environ, SGLANG_PYTHON=sys.executable)
            calls = []
            real_spawn = manager.spawn

            def spawn(command, child_env, log, local_ipc=False):
                calls.append((command[0], child_env["IMAGE_ENGINE"], local_ipc))
                return real_spawn([sys.executable, "-c", "pass"], child_env, log)

            try:
                with patch.object(manager, "spawn", side_effect=spawn):
                    for engine in profiles:
                        manager.run_image(engine, env, Path(directory) / (engine + ".log"))
                        self.assertIsNone(manager.process)
                        self.assertIsNone(manager.active)
                self.assertEqual(calls, [(sys.executable, "diffusers", False), (sys.executable, "sglang", True)])
            finally:
                manager.close()

    def test_switch_reaps_previous_worker_and_reuses_matching_profile(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, UI_ENGINE_PORT=str(port), UI_START_TIMEOUT="5"):
            profiles = {key: {"kind": "chat", "engine": "ninfer", "label": key, "required_env": []} for key in ("a", "b")}
            manager = Manager(profiles, directory)
            real_spawn = manager.spawn
            def spawn(command, env, log):
                code = f"""
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b'{{"data":[{{"id":"fake"}}]}}')
HTTPServer(('127.0.0.1',{port}),H).serve_forever()
"""
                return real_spawn([sys.executable, "-c", code], env, log)
            try:
                with patch.object(manager, "spawn", side_effect=spawn), patch("subprocess.check_output", return_value="fake"):
                    manager.ensure_chat("a", 1024)
                    previous = manager.process
                    self.assertEqual(manager.ensure_chat("a", 1024), 0)
                    self.assertIs(previous, manager.process)
                    manager.ensure_chat("b", 1024)
                    self.assertIsNotNone(previous.poll())
                    self.assertEqual(manager.active, "b")
            finally:
                manager.close()
            self.assertIsNone(manager.process)

    def test_failed_start_is_cleaned_up(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = {"kind": "chat", "engine": "ninfer", "required_env": [], "label": "bad"}
            manager = Manager({"bad": profile}, directory)
            real_spawn = manager.spawn
            def spawn(command, env, log):
                return real_spawn([sys.executable, "-c", "raise SystemExit(1)"], env, log)
            with patch.object(manager, "spawn", side_effect=spawn), patch("subprocess.check_output", return_value="fake"):
                with self.assertRaises(RuntimeError):
                    manager.ensure_chat("bad", 1024)
            self.assertIsNone(manager.process)


class BenchmarkTests(unittest.TestCase):
    def test_matched_payloads_fresh_trials_and_downloads(self):
        class FakeManager:
            registry = {key: {"label": key, "source": ".", "required_env": []} for key in ("a", "b")}
            lock = threading.Lock()
            url, model, last_command = "http://127.0.0.1", "model", "fake"
            def __init__(self, directory):
                self.runtime, self.starts, self.stopped = Path(directory), [], False
            def ensure_chat(self, key, context, fresh):
                self.starts.append((key, context, fresh))
                return 1
            def stop(self):
                self.stopped = True
        bodies = []
        def stream(url, body, timeout):
            bodies.append(body)
            yield {"done": True, "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "timings": {},
                   "ttft": 0.1, "elapsed": 1.0, "tokens_per_second": 4.44, "finish_reason": "length"}
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmarks, "stream", side_effect=stream), patch.object(benchmarks, "capture", return_value="test"):
            manager = FakeManager(directory)
            results = list(benchmarks.run(manager, ["a", "b"], "same prompt", 5, 2, "warm", True, 1024))
            self.assertEqual(len(bodies), 8)
            self.assertTrue(all(body == bodies[0] for body in bodies))
            self.assertTrue(all(start[2] for start in manager.starts))
            self.assertTrue(manager.stopped)
            rows, status, paths = results[-1]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(Path(path).is_file() for path in paths))
            self.assertEqual(json.loads(Path(paths[0]).read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()

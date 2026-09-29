"""Own exactly one inference process group; callers hold the shared GPU lock."""

import json
from contextlib import nullcontext
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from .offline import environment
from .registry import ROOT, availability, profile_env


class Manager:
    def __init__(self, registry, runtime, live_logs=False):
        self.registry, self.runtime = registry, Path(runtime)
        self.lock = threading.Lock()
        self.process = None
        self.active = None
        self.model = None
        self.context = None
        self.configured_port = int(os.environ["UI_ENGINE_PORT"])
        self.port = self.configured_port
        self.url = f"http://127.0.0.1:{self.port}"
        self.closed = False
        self.last_command = None
        self.live_logs = live_logs
        self.image_control = None
        self.image_signature = None
        self.image_progress_path = None

    def status(self):
        if self.process is not None and self.process.poll() is None:
            return f"Loaded: {self.registry[self.active]['label']}"
        return "No model loaded · GPU available"

    def stop(self):
        self.image_progress_path = None
        if self.image_control is not None:
            self.image_control.close()
            self.image_control = None
        self.image_signature = None
        process = self.process
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        self.process = self.active = self.model = self.context = None

    def spawn(self, command, env, log_path, pass_fds=()):
        if self.closed:
            raise RuntimeError("cook-4090 is shutting down")
        env = dict(env, **environment(), PYTHONUNBUFFERED="1")
        # Keep transient library caches and temporary artifacts inside the session.
        env.update(TMPDIR=str(self.runtime), GRADIO_TEMP_DIR=str(self.runtime / "gradio"))
        # Foreground inference inherits the terminal. Benchmarks retain worker logs.
        with (nullcontext(None) if self.live_logs else open(log_path, "w")) as log:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "src.offline", *command], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=pass_fds,
            )
        return self.process

    def ensure_chat(self, key, context, fresh=False):
        if int(context) < 1024:
            raise ValueError("Context capacity must be at least 1024 tokens")
        profile = self.registry[key]
        if profile["kind"] != "chat" or availability(profile) != "Installed":
            raise ValueError("Choose an installed chat model")
        if not fresh and self.active == key and self.context == context and self.process and self.process.poll() is None:
            return 0.0
        self.stop()
        if self.configured_port == 0:
            with socket.socket() as allocation:
                allocation.bind(("127.0.0.1", 0))
                self.port = allocation.getsockname()[1]
            self.url = f"http://127.0.0.1:{self.port}"
        with socket.socket() as probe:
            probe.settimeout(1)
            # A live listener matters here; a bind probe also rejects harmless
            # TIME_WAIT connections left by engines with different socket options.
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"Local engine port {self.port} is already in use; not taking over another server") from None
        env = profile_env(profile)
        env.update(HOST="127.0.0.1", PORT=str(self.port), CONTEXT=str(context))
        command = [str(ROOT / "run.sh"), "serve", profile["engine"], *profile.get("args", [])]
        self.last_command = subprocess.check_output(command, env=dict(env, DRY_RUN="1"), text=True).strip()
        log_path = self.runtime / "engine.log"
        started = time.perf_counter()
        process = self.spawn(command, env, log_path)
        self.active, self.context = key, context
        deadline = time.monotonic() + int(os.environ["UI_START_TIMEOUT"])
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    detail = "See the inference terminal." if self.live_logs else log_path.read_text(errors="replace")[-1800:]
                    raise RuntimeError("Engine failed to start: " + detail)
                try:
                    with urllib.request.urlopen(self.url + "/v1/models", timeout=2) as response:
                        models = json.load(response).get("data", [])
                        if models:
                            self.model = models[0]["id"]
                            return time.perf_counter() - started
                except (OSError, ValueError):
                    pass
                time.sleep(0.5)
            raise TimeoutError("Engine startup timed out")
        except BaseException:
            self.stop()
            raise

    def run_image(self, key, env, log):
        profile = self.registry[key]
        if profile["kind"] != "image" or availability(profile) != "Installed":
            raise ValueError("Choose an installed image model")
        engine = profile["engine"]
        env = dict(env, IMAGE_ENGINE=engine)
        python = sys.executable
        env["PATH"] = str(Path(python).parent) + os.pathsep + env["PATH"]
        reuse = env.get("IMAGE_REUSE", "off") == "on"
        # Changes to pipeline construction need a new worker. Per-request settings
        # (prompt, references, dimensions, steps, seed, tiling) travel over the socket.
        signature = tuple(env.get(name) for name in
                          ("IMAGE_ENGINE", "IMAGE_WEIGHTS", "IMAGE_ATTENTION", "IMAGE_QUANTIZATION",
                           "IMAGE_COMPILE", "IMAGE_FF_CHUNK_SIZE", "IMAGE_DIFFUSERS_REVISION"))
        try:
            if not (reuse and self.active == key and self.image_signature == signature
                    and self.image_control is not None and self.process and self.process.poll() is None):
                self.stop()
                self.active = key
                if reuse:
                    self.image_control, child = socket.socketpair()
                    try:
                        env["IMAGE_WORKER_FD"] = str(child.fileno())
                        self.spawn([python, "-u", "-m", "src.image_worker"], env, log, pass_fds=(child.fileno(),))
                    finally:
                        child.close()
                    self.image_signature = signature
                else:
                    self.spawn([python, "-u", "-m", "src.image_worker"], env, log)
            timeout = int(os.environ["IMAGE_UI_TIMEOUT"])
            self.image_progress_path = Path(env["IMAGE_OUTPUT"]) / "progress.json"
            if reuse:
                request = {name.removeprefix("IMAGE_").lower(): value for name, value in env.items()
                           if name.startswith("IMAGE_") and name != "IMAGE_WORKER_FD"}
                self.image_control.settimeout(timeout)
                self.image_control.sendall((json.dumps(request) + "\n").encode())
                # One-byte acknowledgement avoids framing ambiguity on stream sockets.
                complete = self.image_control.recv(1) == b"\x01"
            else:
                complete = self.process.wait(timeout=timeout) == 0
            if not complete:
                detail = "See the inference terminal." if self.live_logs else "See the image worker log."
                raise RuntimeError("Image generation failed: " + detail)
            if not reuse:
                self.stop()
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.stop()
            raise RuntimeError("Image worker stopped responding; see the inference log.") from exc
        except BaseException:
            self.stop()
            raise
        finally:
            self.image_progress_path = None

    def close(self):
        self.closed = True
        self.stop()

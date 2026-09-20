"""Own exactly one inference process group; callers hold the shared GPU lock."""

import json
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
    def __init__(self, registry, runtime):
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

    def status(self):
        if self.process is not None and self.process.poll() is None:
            return f"Loaded: {self.registry[self.active]['label']}"
        return "No model loaded · GPU available"

    def stop(self):
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

    def spawn(self, command, env, log_path):
        if self.closed:
            raise RuntimeError("cook-4090 is shutting down")
        env = dict(env, **environment(), PYTHONUNBUFFERED="1")
        # Keep transient library caches and temporary artifacts inside the session.
        env.update(TMPDIR=str(self.runtime), GRADIO_TEMP_DIR=str(self.runtime / "gradio"))
        with open(log_path, "w") as log:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "src.offline", *command], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
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
                    raise RuntimeError("Engine failed to start: " + log_path.read_text(errors="replace")[-1800:])
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
        self.stop()
        self.active = key
        process = self.spawn([sys.executable, "-u", "-m", "src.image_worker"], env, log)
        try:
            code = process.wait(timeout=int(os.environ["IMAGE_UI_TIMEOUT"]))
            if code:
                raise RuntimeError("Image generation failed: " + Path(log).read_text(errors="replace")[-1400:])
        finally:
            self.stop()

    def close(self):
        self.closed = True
        self.stop()

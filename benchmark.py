#!/usr/bin/env -S uv run --script
"""Run the fixed 256K comparison and write one report."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent


def default_workload(max_tokens=None):
    source = ROOT / "backends/ninfer/examples/cli/messages/long_niah_256k.json"
    messages = json.loads(source.read_text())
    content = messages[-1]["content"]
    question = content.rfind("What are the ORCHID")
    if question < 0:
        raise RuntimeError(f"Question missing from {source}")
    messages[-1]["content"] = content[:question] + (
        "What are the ORCHID relay recovery code and registered status color? "
        "Respond with exactly: ORCHID=<code>; COLOR=<color>"
    )
    return {
        "name": "long-context-256k",
        "source": str(source.relative_to(ROOT)),
        "max_tokens": max_tokens or int(os.environ.get("BENCHMARK_TOKENS", "32")),
        "expected_text": "ORCHID=493817; COLOR=COBALT",
        "messages": messages,
    }


def observe_stream(lines, started):
    first = None
    usage, timings, text, reasoning, finish = {}, {}, [], [], None
    done = False
    for raw in lines:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if event.get("error"):
            raise RuntimeError(event["error"])
        usage = event.get("usage") or usage
        timings = event.get("timings") or timings
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content") or ""
            thought = delta.get("reasoning_content") or ""
            if (content or thought) and first is None:
                first = time.perf_counter() - started
            text.append(content)
            reasoning.append(thought)
            finish = choice.get("finish_reason") or finish
    if not done or first is None or "completion_tokens" not in usage:
        raise RuntimeError("Incomplete stream")
    total = time.perf_counter() - started
    tokens = usage["completion_tokens"]
    return {
        "client_ttft_seconds": first,
        "client_tpot_seconds": (total - first) / (tokens - 1) if tokens > 1 else None,
        "client_total_seconds": total,
        "usage": usage,
        "server_timings": timings,
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "finish_reason": finish,
    }


def request(url, payload, timeout):
    request_data = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request_data, timeout=timeout) as response:
            return observe_stream(response, started)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def wait_ready(process, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited; inspect its log")
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=2) as response:
                if json.load(response).get("data"):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.25)
    raise TimeoutError("Server startup timed out")


def stop(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def capture(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sanitize_log(path, command):
    if not path.exists():
        return
    text = path.read_text(errors="replace")
    for value in sorted(set(command), key=len, reverse=True):
        candidate = Path(value)
        if not candidate.is_absolute():
            continue
        try:
            replacement = "<repo>/" + str(candidate.relative_to(ROOT))
        except ValueError:
            replacement = "<external>/" + candidate.name
        text = text.replace(value, replacement)
    path.write_text(text)


def main():
    out = ROOT / "results"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir()
    logs = out / "logs"
    logs.mkdir()
    report_path = out / "report.json"

    workload = default_workload()
    warmup = int(os.environ.get("BENCHMARK_WARMUP", "0"))
    repeats = int(os.environ.get("BENCHMARK_REPEATS", "1"))
    timeout = float(os.environ.get("BENCHMARK_TIMEOUT", "600"))
    if warmup < 0 or repeats < 1 or timeout <= 0:
        raise ValueError("Invalid benchmark settings")

    report = {
        "started": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "gpu": capture([
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,memory.total",
            "--format=csv,noheader",
        ]),
        "workload_sha256": hashlib.sha256(
            json.dumps(workload, sort_keys=True).encode()
        ).hexdigest(),
        "workload": {key: value for key, value in workload.items() if key != "messages"},
        "measurement": {
            "warmup_requests": warmup,
            "measured_repeats": repeats,
            "temperature": 0,
            "seed": 42,
            "prompt_cache": False,
        },
        "runs": [],
    }

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for backend in ("ninfer", "llamacpp"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = {
                **os.environ,
                "HOST": "127.0.0.1",
                "PORT": str(port),
                "HOST_STATE_SLOTS": "0",
            }
            command = [str(ROOT / "run.sh"), "serve", backend]
            resolved = subprocess.check_output(
                command, env={**env, "DRY_RUN": "1"}, text=True
            ).strip()
            binary = shlex.split(resolved)[0]
            run = {
                "backend": backend,
                "server_version": capture([binary, "--version"]),
                "server_log": f"logs/{backend}.log",
                "results": [],
            }
            report["runs"].append(run)
            save()
            url = f"http://127.0.0.1:{port}"
            server_argv = shlex.split(resolved)
            log_path = logs / f"{backend}.log"

            try:
                with log_path.open("w") as log:
                    process = subprocess.Popen(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=ROOT,
                    )
                    try:
                        wait_ready(process, url, timeout)
                        with urllib.request.urlopen(url + "/v1/models", timeout=5) as response:
                            model = json.load(response)["data"][0]["id"]
                        payload = {
                            "model": model,
                            "messages": workload["messages"],
                            "stream": True,
                            "stream_options": {"include_usage": True},
                            "max_tokens": workload["max_tokens"],
                            "temperature": 0,
                            "seed": 42,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                        if backend == "llamacpp":
                            payload["cache_prompt"] = False
                        else:
                            payload["prompt_cache_options"] = {"mode": "explicit"}

                        for _ in range(warmup):
                            request(url, payload, timeout)
                        for repeat in range(1, repeats + 1):
                            result = request(url, payload, timeout)
                            result["repeat"] = repeat
                            result["exact_answer"] = (
                                workload["expected_text"] == result["text"].strip()
                            )
                            result["reached_token_limit"] = (
                                result["usage"]["completion_tokens"] == workload["max_tokens"]
                            )
                            run["results"].append(result)
                            save()
                            print(f"DONE: {backend}", flush=True)
                    finally:
                        stop(process)
            finally:
                sanitize_log(log_path, server_argv)

            run["quality_pass"] = all(
                result["exact_answer"] for result in run["results"]
            )
            save()

        report["quality_pass"] = all(run["quality_pass"] for run in report["runs"])
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

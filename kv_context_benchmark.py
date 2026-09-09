"""Incremental native-server KV capacity probes followed by bounded quality checks."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import benchmark as bench
import quality_benchmark as quality

ROOT = Path(__file__).resolve().parent


def startup_record(path):
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") == "server_start":
            return record
    raise RuntimeError("ready server did not record startup memory")


def main():
    start, step, maximum = (int(os.environ[f"KV_CONTEXT_{k}"]) for k in ("START", "STEP", "MAX"))
    margin = int(os.environ["KV_CONTEXT_HEADROOM_MIB"]) * 1024**2
    timeout = int(os.environ["KV_CONTEXT_TIMEOUT"])
    output = int(os.environ["KV_CONTEXT_OUTPUT"])
    seed = int(os.environ["KV_CONTEXT_SEED"])
    reserve = int(os.environ["KV_CONTEXT_INPUT_RESERVE"])
    scale = float(os.environ["KV_CONTEXT_INPUT_SCALE"])
    modes = os.environ["KV_CONTEXT_MODES"].split(",")
    if (step <= 0 or start <= output + reserve or maximum < start or margin < 0 or
            scale <= 0 or not modes or any(m not in ("fp8", "int8") for m in modes)):
        raise ValueError("invalid KV context experiment settings")
    out = ROOT / "results/quality" / ("kv-context-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True)
    report = dict(status="running", settings={k: v for k, v in os.environ.items() if k.startswith("KV_CONTEXT_")},
                  weights=os.environ["VISION_BF16_WEIGHTS"], cache="prefix reuse disabled; media preprocessing may warm",
                  gpu=bench.capture(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,memory.total", "--format=csv,noheader"]),
                  probes=[], profiles=[], records=[], practical_capacity={},
                  interpretation="Startup fit is distinct from quality; single-seed synthetic diagnostics, not general reasoning qualification")
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Results:", out.relative_to(ROOT), flush=True)
    save()

    def launch(kv, capacity, phase):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        label = f"{phase}-{kv}-{capacity}"
        env = dict(os.environ, HOST="127.0.0.1", PORT=str(port), CONTEXT=str(capacity),
                   NINFER_WEIGHTS=os.environ["VISION_BF16_WEIGHTS"], VISION="on", NINFER_VISION_OFFLOAD="on",
                   NINFER_KV=kv, NINFER_SPEC=os.environ["KV_CONTEXT_SPEC"],
                   NINFER_DRAFT=os.environ["KV_CONTEXT_DRAFT"], NINFER_MTP_ADAPTIVE="off",
                   NINFER_PREFIX_REUSE="off", NINFER_CHUNK=os.environ["KV_CONTEXT_CHUNK"],
                   NINFER_VISION_TOKENS=os.environ["KV_CONTEXT_VISION_TOKENS"])
        log_path, native = out / f"{label}.log", out / f"{label}-requests.jsonl"
        cmd = [str(ROOT / "run.sh"), "serve", "ninfer", "--request-log-jsonl", str(native)]
        row = dict(kv=kv, capacity=capacity, phase=phase, status="starting",
                   command=subprocess.check_output(cmd, env=dict(env, DRY_RUN="1"), text=True).strip())
        report["probes" if phase == "probe" else "profiles"].append(row)
        save()
        log = log_path.open("w")
        server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        url = f"http://127.0.0.1:{port}"
        try:
            bench.wait_ready(server, url, timeout)
            record = startup_record(native)
            row.update(status="ready", memory=record["memory"], engine=record["engine"], artifact=record["artifact"])
            row["meets_headroom"] = row["memory"]["available_after_startup_bytes"] >= margin
            save()
            return server, log, url, row
        except BaseException as error:
            bench.stop(server)
            log.close()
            text = log_path.read_text()
            # Only an explicit memory rejection establishes a capacity boundary.
            memory_rejection = any(s in text.lower() for s in (
                "out of memory", "cannot fit", "exceeds available", "insufficient device memory",
                "requested engine runtime reservation requires", "model weights require"))
            row.update(status="memory_rejected" if memory_rejection else "startup_failed",
                       error=str(error), log_tail=text[-4000:])
            save()
            if phase == "probe" and memory_rejection and isinstance(error, Exception):
                return None, None, url, row
            raise

    with (out / "gpu.csv").open("w") as gpu_log:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu", "--format=csv", "-l", "1"], stdout=gpu_log)
        try:
            for kv in modes:
                for capacity in range(start, maximum + 1, step):
                    server, log, _, row = launch(kv, capacity, "probe")
                    if server is None:
                        print(kv, capacity, row["status"], flush=True)
                        break
                    try:
                        free = row["memory"]["available_after_startup_bytes"]
                        print(kv, capacity, "startup free MiB", round(free / 1024**2, 1),
                              "practical", row["meets_headroom"], flush=True)
                        row["status"] = "complete"
                        if row["meets_headroom"]:
                            report["practical_capacity"][kv] = capacity
                    finally:
                        bench.stop(server)
                        log.close()
                        save()
                    if not row["meets_headroom"]:
                        break

            targets = [(kv, start) for kv in modes if kv in report["practical_capacity"]]
            targets += [(kv, report["practical_capacity"][kv]) for kv in modes
                        if report["practical_capacity"].get(kv, start) > start]
            for kv, capacity in targets:
                server, log, url, profile = launch(kv, capacity, "quality")
                try:
                    if not profile["meets_headroom"]:
                        raise RuntimeError("quality startup lost required VRAM headroom")
                    # Character-based estimate only. Actual template/media-expanded token
                    # counts and finish reasons come from the server for every request.
                    nominal = int((capacity - output - reserve) * scale)
                    cases = list(quality.workloads(capacity, seed, nominal_input=nominal))
                    for case in cases:
                        if case["label"] == "ledger-reasoning":
                            case["max_tokens"] = output
                    fixtures = out / f"workloads-{capacity}.json"
                    fixture_bytes = (json.dumps(cases, indent=2) + "\n").encode()
                    if fixtures.exists() and fixtures.read_bytes() != fixture_bytes:
                        raise RuntimeError("paired workloads differ")
                    fixtures.write_bytes(fixture_bytes)
                    profile["workloads_sha256"] = hashlib.sha256(fixture_bytes).hexdigest()
                    for case in cases:
                        payload = dict(model="qwen3.8-27b", messages=case["messages"], max_tokens=case["max_tokens"],
                                       temperature=0, seed=seed, top_p=1, top_k=0, min_p=0,
                                       presence_penalty=0, frequency_penalty=0, reasoning_effort=case["effort"],
                                       stream=True, stream_options={"include_usage": True})
                        result = quality.request(url, payload, timeout)
                        if result["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0):
                            raise RuntimeError("unexpected prefix reuse")
                        row = dict(kv=kv, capacity=capacity, fixture=case["label"], expected=case["expected"],
                                   max_tokens=case["max_tokens"], effort=case["effort"],
                                   score=quality.score(result["text"], case["expected"]), **result)
                        report["records"].append(row)
                        save()
                        print(kv, capacity, case["label"], "prompt", result["usage"]["prompt_tokens"],
                              "output", result["usage"]["completion_tokens"], row["score"],
                              "finish", result["finish_reason"], flush=True)
                    profile["status"] = "complete"
                except BaseException as error:
                    profile.update(status="failed", error=str(error))
                    raise
                finally:
                    bench.stop(server)
                    log.close()
                    save()
            report["status"] = "complete" if len(report["practical_capacity"]) == len(modes) else "no_practical_capacity"
        except BaseException as error:
            report.update(status="failed", error=str(error) or type(error).__name__)
            raise
        finally:
            bench.stop(monitor)
            save()


if __name__ == "__main__":
    main()

"""Matched UI benchmark workloads; exports are disposable until downloaded."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import uuid

from .client import payload, stream
from .registry import ROOT, profile_env

HEADERS = ["Model / engine", "Trial", "Prompt tokens", "Output tokens", "Cached tokens", "TTFT (s)", "Output tok/s", "Total (s)", "Finish"]


def capture(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def identity(profile):
    env = profile_env(profile)
    artifacts = {}
    for key in profile["required_env"]:
        from pathlib import Path
        path = Path(env[key])
        stat = path.stat()
        artifacts[key] = {"path": os.path.relpath(path, ROOT), "size_bytes": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns}
    return {"profile": profile, "artifacts": artifacts,
            "source_revision": capture(["git", "-C", str(ROOT / profile["source"]), "rev-parse", "HEAD"])}


def run(manager, keys, prompt, tokens, repeats, cache, thinking, context):
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Select at least one model, without duplicates")
    if not prompt.strip() or len(prompt) > 100000:
        raise ValueError("Enter a workload of at most 100,000 characters")
    if not 1 <= int(tokens) < int(context) or not 1 <= int(repeats) <= 10 or cache not in {"cold", "warm"}:
        raise ValueError("Invalid benchmark settings")
    output = manager.runtime / ("benchmark-" + uuid.uuid4().hex)
    output.mkdir()
    report = {"created": datetime.now(timezone.utc).isoformat(), "status": "running",
              "settings": {"max_tokens": int(tokens), "repeats": int(repeats), "cache": cache,
                           "thinking": bool(thinking), "context": int(context), "temperature": 0,
                           "seed": int(os.environ["UI_BENCH_SEED"]), "stream": True},
              "messages": [{"role": "user", "content": prompt}],
              "workload_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "cache_policy": "Fresh engine for every trial; warm mode runs the exact workload once before measuring. OS file cache uncontrolled.",
              "gpu": capture(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,power.limit", "--format=csv"]),
              "metric_notes": "Client TTFT includes reasoning tokens. Output rate is (completion_tokens-1)/(total-TTFT); missing engine usage stays null. Model load is excluded. These are deployment comparisons, not identical quantizations.",
              "engines": {}, "records": []}
    rows = []
    paths = [str(output / "report.json"), str(output / "summary.csv")]

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        with (output / "summary.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)
            writer.writerows(rows)

    save()
    with manager.lock:
        try:
            for key in keys:
                profile = manager.registry[key]
                report["engines"][key] = identity(profile)
                for repeat in range(int(repeats)):
                    yield rows, f"Loading {profile['label']} · trial {repeat + 1}/{repeats}", paths
                    load_seconds = manager.ensure_chat(key, int(context), fresh=True)
                    report["engines"][key]["command"] = manager.last_command
                    body = payload(manager.model, report["messages"], tokens, 0, thinking, os.environ["UI_BENCH_SEED"])
                    body.update(top_p=1, top_k=0, min_p=0, presence_penalty=0, frequency_penalty=0)
                    warmup = None
                    if cache == "warm":
                        yield rows, f"Warming exact workload · {profile['label']}", paths
                        for event in stream(manager.url, body, int(os.environ["UI_REQUEST_TIMEOUT"])):
                            warmup = event
                    yield rows, f"Measuring {profile['label']} · trial {repeat + 1}/{repeats}", paths
                    result = None
                    for event in stream(manager.url, body, int(os.environ["UI_REQUEST_TIMEOUT"])):
                        result = event
                    if not result or not result["done"]:
                        raise RuntimeError("Incomplete benchmark stream")
                    usage = result["usage"]
                    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", result["timings"].get("cache_n"))
                    report["records"].append(dict(profile=key, trial=repeat + 1, load_seconds=load_seconds,
                                                  warmup=warmup, cached_tokens=cached, **result))
                    rows.append([profile["label"], repeat + 1, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                                 cached, result["ttft"], result["tokens_per_second"], result["elapsed"], result["finish_reason"]])
                    save()
                    yield rows, f"Completed {len(rows)} of {len(keys) * int(repeats)} trials", paths
            report["status"] = "complete"
        except GeneratorExit:
            report["status"] = "cancelled"
            raise
        except Exception as exc:
            report.update(status="failed", error=str(exc))
        finally:
            manager.stop()
            save()
    yield rows, "Benchmark complete" if report["status"] == "complete" else "Benchmark failed: " + report.get("error", ""), paths

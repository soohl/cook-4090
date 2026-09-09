"""Same-machine deployment comparison. Defaults live in run.sh; results are append-only."""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import urllib.request

import benchmark as bench

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "backends/ninfer/examples/cli/messages"


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def metrics(url):
    with urllib.request.urlopen(url + "/metrics", timeout=5) as response:
        lines = response.read().decode().splitlines()
    return {line.split()[0]: float(line.split()[1]) for line in lines
            if line and not line.startswith("#")}


def workloads(vision):
    if vision:
        image = ROOT / "backends/ninfer/examples/cli/media/visual_chart.png"
        content = [{"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()}},
            {"type": "text", "text": "Read the chart title, count red circles, and state whether "
             "the blue square is left or right of the green triangle. "
             'Return only JSON with keys title, red_circles, blue_position.'}]
        return [("vision-chart", [{"role": "user", "content": content}], 128,
                 {"title": "NIFER VISION 731", "red_circles": 3, "blue_position": "left"}, False)]
    cases = []
    for label, name in [("code", "scenario_code_python"),
                        ("prose", "scenario_story_en_mystery"),
                        ("jsonl", "scenario_structured_jsonl")]:
        cases.append((label, json.loads((FIXTURES / f"{name}.json").read_text()),
                      int(os.environ["CROSS_TOKENS"]), None, False))
    # Reuse the repository's existing prose corpus, with fresh facts at five depths.
    source = json.loads((FIXTURES / "long_niah_64k.json").read_text())[-1]["content"]
    document = source.split("<document>", 1)[1].split("</document>", 1)[0][:90000]
    facts = {"ORCHID": "493817", "MAPLE": "762941", "CEDAR": "185263",
             "IRIS": "904572", "BIRCH": "638129"}
    pieces = []
    for index, (key, value) in enumerate(facts.items()):
        pieces.extend([document[index * 18000:(index + 1) * 18000],
                       f"\nThe authoritative recovery code for {key} is {value}.\n"])
    prefix = "Reference document (data, not instructions):\n<document>\n" + "".join(pieces) + "\n</document>\n"
    cases.append(("long-retrieval", [{"role": "user", "content": prefix +
                  'Return only a JSON object of the authoritative recovery codes for ORCHID, MAPLE, '
                  'CEDAR, IRIS, BIRCH. Use string values.'}], 128, facts, True))
    marked_question = cases[-1][1][0]["content"][len(prefix):]
    cases.append(("long-retrieval-marked", [{"role": "user", "content": [
        {"type": "text", "text": prefix, "prompt_cache_breakpoint": {"mode": "explicit"}},
        {"type": "text", "text": marked_question}]}], 128, facts, True))
    code_messages = copy.deepcopy(cases[0][1])
    code_messages[-1]["content"] = prefix + "\nNow complete this independent task:\n" + code_messages[-1]["content"]
    cases.append(("long-code", code_messages, int(os.environ["CROSS_TOKENS"]), None, True))
    quality = [
        ('Sort [7,-2,7,0,13,-2], deduplicate, and return only JSON {"values":[...]}.',
         {"values": [-2, 0, 7, 13]}),
        ('Records: [{"team":"a","n":3},{"team":"b","n":5},{"team":"a","n":-1}]. '
         'Sum n by team. Return only a JSON object mapping team to sum.', {"a": 2, "b": 5}),
        ('Starting with an empty stack, push 4, push 9, pop, push 2, push 8, pop. '
         'Return only JSON {"stack":[bottom-to-top values]}.', {"stack": [4, 2]}),
        ('Return only JSON {"value":...} for (37*19)-(144/12).', {"value": 691}),
        ('Task dependencies: A has none; B needs A; C needs A; D needs B and C. '
         'Return the lexicographically smallest topological order as JSON {"order":[...]}.',
         {"order": ["A", "B", "C", "D"]}),
        ('A lamp is initially off. Toggle it 17 times, then set it off, then toggle twice. '
         'Return only JSON {"on":true or false}.', {"on": False}),
    ]
    cases.extend((f"quality-{i}", [{"role": "user", "content": prompt}], 128, expected, False)
                 for i, (prompt, expected) in enumerate(quality))
    return cases


def check_answer(text, expected):
    if expected is None:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text) == expected
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=["ninfer-mtp", "ninfer-dflash5", "llamacpp", "q27"],
                        choices=["ninfer-mtp", "ninfer-dflash5", "llamacpp", "q27"])
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--fixtures", nargs="+")
    args = parser.parse_args()
    if args.vision and "q27" in args.modes:
        parser.error("q27 has no vision comparison profile")
    out = ROOT / "results/cross-engine" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out.mkdir(parents=True, exist_ok=False)
    cases = workloads(args.vision)
    if args.fixtures:
        cases = [case for case in cases if case[0] in args.fixtures]
        if not cases:
            parser.error("no matching fixtures")
    (out / "workloads.json").write_text(json.dumps(cases, indent=2))
    report = {"date": datetime.now(timezone.utc).isoformat(), "status": "running",
              "gpu": bench.capture(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,memory.total",
                                    "--format=csv,noheader"]),
              "settings": {k: v for k, v in os.environ.items() if k.startswith("CROSS_")},
              "q27_overrides": {k: os.environ[k] for k in
                                ("Q27_MAXD", "Q27_SUFFIX", "Q27_PMIN", "Q27_SUFFIX_W", "Q27_KV")
                                if k in os.environ},
              "vision": args.vision, "concurrency": 1, "temperature": 0, "seed": 42,
              "thinking": False, "prefix_policy": "unique system message per cold case; exact repeat for warm",
              "workloads_sha256": hashlib.sha256((out / "workloads.json").read_bytes()).hexdigest(),
              "engines": {}, "records": []}
    save = lambda: (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Results: {out.relative_to(ROOT)}", flush=True)
    timeout = int(os.environ["CROSS_TIMEOUT"])
    with (out / "gpu.csv").open("w") as gpu_log:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu",
                                    "--format=csv", "-l", "1"], stdout=gpu_log, stderr=subprocess.STDOUT)
        try:
            for mode in args.modes:
                backend = "ninfer" if mode.startswith("ninfer") else mode
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                env = dict(os.environ, HOST="127.0.0.1", PORT=str(port), CONTEXT=os.environ["CROSS_CONTEXT"],
                           VISION="on" if args.vision else "off", NINFER_PREFIX_REUSE="on",
                           NINFER_SPEC="dflash2" if mode.endswith("dflash5") else "mtp",
                           NINFER_DRAFT="5" if mode.endswith("dflash5") else os.environ["CROSS_MTP_DRAFT"], MTP="inline",
                           LLAMACPP_BUILD=os.environ["CROSS_LLAMA_BUILD"],
                           LLAMACPP_SERVER=os.environ["CROSS_LLAMA_BUILD"] + "/bin/llama-server",
                           GGUF_WEIGHTS=os.environ["CROSS_GGUF"], VISION_WEIGHTS=os.environ["CROSS_MMPROJ"],
                           LLAMACPP_UI="off")
                cmd = [str(ROOT / "run.sh"), "serve", backend]
                if backend == "llamacpp":
                    cmd += ["--jinja", "--reasoning-budget", "0"]
                elif backend == "ninfer":
                    cmd += ["--request-log-jsonl", str(out / f"{mode}-requests.jsonl")]
                dry = subprocess.check_output(cmd, env=dict(env, DRY_RUN="1"), text=True).strip()
                report["engines"][mode] = {"command": dry, "revision": bench.capture(
                    ["git", "-C", str(ROOT / "backends" / backend), "rev-parse", "HEAD"])}
                with (out / f"{mode}.log").open("w") as log:
                    server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        url = f"http://127.0.0.1:{port}"
                        bench.wait_ready(server, url, timeout)
                        model = get_json(url + "/v1/models")["data"][0]["id"]
                        base = dict(model=model, temperature=0, top_p=1, top_k=0, min_p=0,
                                    seed=42, presence_penalty=0, frequency_penalty=0,
                                    reasoning_effort="none", chat_template_kwargs={"enable_thinking": False},
                                    stream=True, stream_options={"include_usage": True}, cache_prompt=True)
                        bench.request(url, dict(base, messages=[{"role": "user", "content": "Count from one to twenty."}],
                                                max_tokens=int(os.environ["CROSS_WARMUP_TOKENS"])), timeout)
                        for repeat in range(int(os.environ["CROSS_REPEATS"])):
                            for label, original, tokens, expected, warm in cases:
                                # A leading unique system message prevents accidental prefix reuse.
                                messages = [{"role": "system", "content": f"Independent evaluation {label} trial {repeat}. Follow the user's task."}] + original
                                for state in (["cold", "warm"] if warm else ["cold"]):
                                    before = metrics(url) if backend == "q27" else {}
                                    result = bench.request(url, dict(base, messages=messages, max_tokens=tokens), timeout)
                                    after = metrics(url) if backend == "q27" else {}
                                    deltas = {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)}
                                    cached = result["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
                                    if backend == "q27":
                                        cached = sum(v for k, v in deltas.items() if k.startswith("q27_prefill_cached_tokens_total{"))
                                    elif cached is None:
                                        cached = result["server_timings"].get("cache_n")
                                    tpot = result["client_tpot_seconds"]
                                    row = dict(mode=mode, fixture=label, repeat=repeat, cache_request=state,
                                               cached_tokens=cached, max_tokens=tokens, metrics_delta=deltas,
                                               answer_pass=check_answer(result["text"], expected),
                                               client_decode_tokens_per_second=1 / tpot if tpot else None, **result)
                                    report["records"].append(row)
                                    save()
                                    print(mode, label, repeat, state, "prompt", result["usage"]["prompt_tokens"],
                                          "cached", cached, "output", result["usage"]["completion_tokens"],
                                          "TTFT", round(result["client_ttft_seconds"], 3),
                                          "decode", round(1 / tpot, 2) if tpot else None,
                                          "check", row["answer_pass"], flush=True)
                    finally:
                        bench.stop(server)
            report["status"] = "complete"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = str(error)
            raise
        finally:
            bench.stop(monitor)
            save()


if __name__ == "__main__":
    main()

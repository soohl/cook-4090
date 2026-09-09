"""Paired, append-only HTTP vision precision or host-offload experiment on one RTX 4090."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess

from PIL import Image, ImageDraw, ImageFont
import benchmark as bench
from quality_benchmark import request

ROOT = Path(__file__).resolve().parent


def fixtures(out):
    font_path = Path(os.environ["VISION_QUALITY_FONT"])
    font = lambda size: ImageFont.truetype(str(font_path), size)
    cases = []
    for seed in (71, 103):
        rng = random.Random(seed)
        rows = [{"id": "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=8)),
                 "quantity": rng.randint(1, 99), "price": f"{rng.randint(101, 9999) / 100:.2f}"}
                for _ in range(8)]
        for size in (10, 14, 20):
            im = Image.new("RGB", (1280, 960), "white")
            draw = ImageDraw.Draw(im)
            draw.text((60, 50), "STOCK AUDIT / ORIGINAL RECORD", font=font(28), fill="black")
            draw.text((60, 110), "Transcribe every row exactly. Prices have two decimal places.", font=font(18), fill="black")
            for x, title in zip((80, 480, 780), ("ITEM ID", "QUANTITY", "UNIT PRICE")):
                draw.text((x, 180), title, font=font(18), fill="black")
            for i, row in enumerate(rows):
                y = 230 + i * 75
                draw.line((60, y + 45, 1120, y + 45), fill="#cccccc")
                for x, value in zip((80, 480, 780), row.values()):
                    draw.text((x, y), str(value), font=font(size), fill="black")
            path = out / f"ocr-{seed}-{size}px.png"
            im.save(path)
            cases.append(dict(name=path.stem, image=path, expected={"rows": rows},
                              question='Transcribe all eight table rows, in order. Return only JSON: '
                              '{"rows":[{"id":"...","quantity":1,"price":"1.23"},...]}. '
                              'IDs and prices must be strings. Do not calculate totals.'))
    for seed in (211, 313):
        rng = random.Random(seed)
        values = dict(zip(("Aster", "Birch", "Cedar", "Dahlia", "Elm"), rng.sample(range(10, 100, 10), 5)))
        im = Image.new("RGB", (1280, 960), "white")
        draw = ImageDraw.Draw(im)
        draw.text((65, 60), "WAREHOUSE COUNTS", font=font(30), fill="black")
        for tick in range(0, 101, 10):
            x = 220 + tick * 9
            draw.line((x, 180, x, 770), fill="#bbbbbb")
            draw.text((x - 10, 795), str(tick), font=font(20), fill="black")
        for i, (name, value) in enumerate(values.items()):
            y = 220 + i * 110
            draw.text((65, y + 8), name, font=font(24), fill="black")
            draw.rectangle((220, y, 220 + value * 9, y + 45), fill="#3c75b5")
        path = out / f"chart-{seed}.png"
        im.save(path)
        cases.append(dict(name=path.stem, image=path, expected=values,
                          question='Read the horizontal bar chart. Return only a JSON object mapping '
                          'Aster, Birch, Cedar, Dahlia and Elm to their integer counts. Values are multiples of ten.'))
    cases.append(dict(name="basic-vision", image=ROOT / "backends/ninfer/examples/cli/media/visual_chart.png",
                      expected=dict(title="NIFER VISION 731", red_circles=3, blue_position="left"),
                      question='Return only JSON with title (exact image title), red_circles (integer count), '
                      'and blue_position (left or right relative to green triangle).'))
    for case in cases:
        data = case.pop("image").read_bytes()
        case["image_sha256"] = hashlib.sha256(data).hexdigest()
        case["messages"] = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
            {"type": "text", "text": case.pop("question")}]}]
    return cases


def score(text, expected):
    clean = text.strip()
    if clean.startswith("```json") and clean.endswith("```"):
        clean = clean[7:-3].strip()
    try:
        actual = json.loads(clean)
    except ValueError:
        actual = None
    def leaves(value, path=()):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from leaves(item, path + (key,))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                yield from leaves(item, path + (i,))
        else:
            yield path, value
    got = dict(leaves(actual))
    wanted = dict(leaves(expected))
    def exact_match(actual, expected):
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(
                exact_match(actual[k], v) for k, v in expected.items())
        if isinstance(expected, list):
            return len(actual) == len(expected) and all(
                exact_match(a, e) for a, e in zip(actual, expected))
        return actual == expected
    return dict(exact=exact_match(actual, expected), correct=sum(type(got.get(k)) is type(v) and got.get(k) == v for k, v in wanted.items()),
                total=len(wanted), actual=actual)


def main():
    if int(os.environ["VISION_QUALITY_REPEATS"]) < 1 or int(os.environ["VISION_QUALITY_OUTPUT"]) < 1:
        raise ValueError("vision comparison requires positive repeats and output allowance")
    out = ROOT / "results/quality" / ("vision-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    out.mkdir(parents=True)
    cases = fixtures(out)
    (out / "fixtures.json").write_text(json.dumps(cases, indent=2))
    report = dict(status="running", profiles=[], records=[], cache="disabled",
                  settings={k: v for k, v in os.environ.items() if k.startswith("VISION_QUALITY_")},
                  gpu=bench.capture(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,memory.total", "--format=csv,noheader"]),
                  fixtures_sha256=hashlib.sha256((out / "fixtures.json").read_bytes()).hexdigest(),
                  scope="Synthetic visual transcription/chart diagnostic, not OCRBench or a broad quality claim")
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Results:", out.relative_to(ROOT), flush=True)
    save()
    timeout = int(os.environ["VISION_QUALITY_TIMEOUT"])
    comparison = os.environ["VISION_QUALITY_COMPARISON"]
    if comparison == "offload":
        profiles = (("bf16", os.environ["VISION_BF16_WEIGHTS"], "off"),
                    ("bf16-offload", os.environ["VISION_BF16_WEIGHTS"], "on"))
    elif comparison == "precision":
        profiles = (("quantized", os.environ["NINFER_WEIGHTS"], "off"),
                    ("bf16", os.environ["VISION_BF16_WEIGHTS"], "off"))
    else:
        raise ValueError("VISION_QUALITY_COMPARISON must be precision or offload")
    with (out / "gpu.csv").open("w") as gpu_log:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,memory.used,utilization.gpu", "--format=csv", "-l", "1"], stdout=gpu_log)
        try:
            for label, weights, offload in profiles:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                env = dict(os.environ, HOST="127.0.0.1", PORT=str(port), CONTEXT=os.environ["VISION_QUALITY_CONTEXT"],
                           VISION="on", NINFER_WEIGHTS=weights, NINFER_KV=os.environ["VISION_QUALITY_KV"],
                           NINFER_SPEC=os.environ["VISION_QUALITY_SPEC"], NINFER_DRAFT=os.environ["VISION_QUALITY_DRAFT"], NINFER_MTP_ADAPTIVE="off",
                           NINFER_PREFIX_REUSE="off", NINFER_VISION_TOKENS=os.environ["VISION_QUALITY_TOKENS"],
                           NINFER_VISION_OFFLOAD=offload)
                cmd = [str(ROOT / "run.sh"), "serve", "ninfer", "--request-log-jsonl", str(out / f"{label}-requests.jsonl")]
                profile = dict(name=label, command=subprocess.check_output(cmd, env=dict(env, DRY_RUN="1"), text=True).strip(),
                               weights=weights, status="starting")
                report["profiles"].append(profile)
                save()
                with (out / f"{label}.log").open("w") as log:
                    server = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        url = f"http://127.0.0.1:{port}"
                        bench.wait_ready(server, url, timeout)
                        for rep in range(int(os.environ["VISION_QUALITY_REPEATS"])):
                            for case in cases:
                                payload = dict(model="qwen3.8-27b", messages=case["messages"], max_tokens=int(os.environ["VISION_QUALITY_OUTPUT"]),
                                               temperature=0, seed=int(os.environ["VISION_QUALITY_SEED"]), top_p=1, top_k=0, min_p=0, presence_penalty=0,
                                               frequency_penalty=0, reasoning_effort="none", stream=True, stream_options={"include_usage": True})
                                result = request(url, payload, timeout)
                                if result["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0):
                                    raise RuntimeError("unexpected prefix reuse")
                                row = dict(profile=label, repeat=rep, fixture=case["name"], expected=case["expected"],
                                           score=score(result["text"], case["expected"]), **result)
                                report["records"].append(row)
                                save()
                                print(label, rep, case["name"], row["score"]["correct"], "/", row["score"]["total"], result["usage"], flush=True)
                        profile["status"] = "complete"
                    except BaseException:
                        profile["status"] = "failed"
                        raise
                    finally:
                        bench.stop(server)
                        save()
            if comparison == "offload":
                resident = {(r["repeat"], r["fixture"]): r for r in report["records"] if r["profile"] == "bf16"}
                report["exact_response_parity"] = all(
                    r["text"] == resident[r["repeat"], r["fixture"]]["text"]
                    for r in report["records"] if r["profile"] == "bf16-offload")
                if not report["exact_response_parity"]:
                    raise RuntimeError("offload response parity failed; inspect paired records")
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

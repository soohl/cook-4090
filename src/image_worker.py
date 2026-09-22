"""Isolated image worker shared by the CLI and UI; defaults live in run.sh."""

import json
import os
from pathlib import Path
import resource
import subprocess
import time

from .offline import restrict_ui_network

# This module is imported again by SGLang's spawned workers.
restrict_ui_network()

import diffusers
import torch
import transformers
from PIL import Image, ImageOps


def main():
    config = {key.removeprefix("IMAGE_").lower(): value for key, value in os.environ.items() if key.startswith("IMAGE_")}
    engine_name = config["engine"]
    if engine_name not in {"diffusers", "sglang"}:
        raise ValueError("IMAGE_ENGINE must be diffusers or sglang")
    width, height, steps, seed = (int(config[key]) for key in ("width", "height", "steps", "seed"))
    if width <= 0 or height <= 0 or width % 32 or height % 32 or steps <= 0:
        raise ValueError("Dimensions must be positive multiples of 32; steps must be positive")
    tiling = config["vae_tiling"]
    if tiling not in {"auto", "on", "off"}:
        raise ValueError("IMAGE_VAE_TILING must be auto, on, or off")
    tile_size, tile_stride = int(config["vae_tile_size"]), int(config["vae_tile_stride"])
    if not 0 < tile_stride < tile_size or tile_size % 32 or tile_stride % 32:
        raise ValueError("VAE tile size/stride must be multiples of 32, with 0 < stride < size")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "config": config, "engine": engine_name, "dtype": "bfloat16",
        "batch_size": 1, "true_cfg_scale": 1.0, "use_kv_cache": True,
        "cache_state": "fresh process; local weights; no warmup; OS file cache uncontrolled",
        "torch": torch.__version__, "diffusers": diffusers.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip(),
        "gpu_before": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,power.limit", "--format=csv"], text=True).strip(),
    }
    started = time.perf_counter()
    engine = None
    try:
        paths = json.loads(config.get("references", "[]"))
        if not isinstance(paths, list) or len(paths) > int(config["max_references"]):
            raise ValueError("Invalid reference-image list")
        references = []
        report["references"] = []
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                oriented = ImageOps.exif_transpose(source)
                reference = oriented.convert("RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB")
            saved = output / f"reference-{index + 1}.png"
            reference.save(saved)
            references.append(reference)
            report["references"].append({"file": saved.name, "size": list(reference.size), "mode": reference.mode})
        report["mode"] = "image_edit" if references else "text_to_image"
        print(f"Loading local BF16 weights with {engine_name}", flush=True)
        report["image_latent_tokens"] = (width // 16) * (height // 16)
        report["vae_tiling"] = tiling == "on" or (tiling == "auto" and width * height > 1024**2)
        if engine_name == "diffusers":
            from .image_diffusers import DiffusersEngine
            engine = DiffusersEngine(config, report)
        else:
            from .image_sglang import SGLangEngine
            engine = SGLangEngine(config, report)
        report["load_seconds"] = time.perf_counter() - started
        if engine_name == "diffusers":
            torch.cuda.reset_peak_memory_stats()
        generation_start = time.perf_counter()
        image = engine.generate(config, references, report)
        if engine_name == "diffusers":
            torch.cuda.synchronize()
        report["generation_seconds"] = time.perf_counter() - generation_start
        if image.size != (width, height):
            raise RuntimeError(f"Requested {width}x{height}, but pipeline returned {image.size}")
        image.save(output / "image.png")
        report.update(status="success", image_size=list(image.size), image_mode=image.mode)
    except Exception as exc:
        report.update(status="error", error=repr(exc))
        raise
    finally:
        if engine is not None:
            engine.close()
        report["total_seconds"] = time.perf_counter() - started
        if engine_name == "diffusers":
            report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_cuda_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["peak_process_rss_gib"] = max(resource.getrusage(who).ru_maxrss for who in
                                               (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)) / 2**20
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

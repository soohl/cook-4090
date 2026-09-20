"""Isolated Qwen-Image-2.1 worker and CLI benchmark; configuration lives in run.sh."""

import json
import os
from pathlib import Path
import resource
import subprocess
import time

import diffusers
import torch
import transformers
from diffusers import QwenImage21Pipeline
from PIL import Image, ImageOps


def main():
    config = {key.removeprefix("IMAGE_").lower(): value for key, value in os.environ.items() if key.startswith("IMAGE_")}
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
        "config": config, "dtype": "bfloat16", "offload": "model_cpu",
        "batch_size": 1, "true_cfg_scale": 1.0, "use_kv_cache": True,
        "cache_state": "fresh process; local weights; no warmup; OS file cache uncontrolled",
        "torch": torch.__version__, "diffusers": diffusers.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_before": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,power.limit", "--format=csv"], text=True).strip(),
    }
    started = time.perf_counter()
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
        print("Loading local BF16 weights with model CPU offload", flush=True)
        pipe = QwenImage21Pipeline.from_pretrained(config["weights"], torch_dtype=torch.bfloat16, local_files_only=True)
        def record_encoder_tokens(module, args, kwargs):
            report["encoder_input_tokens"] = int(kwargs["attention_mask"].sum())
            report["conditioning_tokens"] = report["encoder_input_tokens"] - pipe._drop_idx

        pipe.text_encoder.register_forward_pre_hook(record_encoder_tokens, with_kwargs=True)
        report["image_latent_tokens"] = (width // pipe.vae_scale_factor) * (height // pipe.vae_scale_factor)
        report["vae_tiling"] = tiling == "on" or (tiling == "auto" and width * height > 1024**2)
        if report["vae_tiling"]:
            pipe.vae.enable_tiling(
                tile_sample_min_height=tile_size, tile_sample_min_width=tile_size,
                tile_sample_stride_height=tile_stride, tile_sample_stride_width=tile_stride,
            )
        pipe.enable_model_cpu_offload()
        report["load_seconds"] = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        generation_start = time.perf_counter()
        image = pipe(
            prompt=config["prompt"], width=width, height=height,
            image=references or None, output_resolution=int(config["reference_resolution"]),
            num_inference_steps=steps, true_cfg_scale=1.0, use_kv_cache=True,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
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
        report["total_seconds"] = time.perf_counter() - started
        report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["peak_cuda_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

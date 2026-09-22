"""Native BF16 SGLang adapter; reuse the original local checkpoint."""

import dataclasses
import importlib.metadata
import os
from pathlib import Path

from PIL import Image


class SGLangEngine:
    def __init__(self, config, report):
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import QwenImage21PipelineConfig
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

        pipeline = QwenImage21PipelineConfig(generator_device="cuda", vae_tiling=report["vae_tiling"])
        pipeline.vae_config.tile_sample_min_height = int(config["vae_tile_size"])
        pipeline.vae_config.tile_sample_min_width = int(config["vae_tile_size"])
        pipeline.vae_config.tile_sample_stride_height = int(config["vae_tile_stride"])
        pipeline.vae_config.tile_sample_stride_width = int(config["vae_tile_stride"])
        self.generator = DiffGenerator.from_pretrained(
            model_path=config["weights"], model_id="Qwen-Image-2.1", backend="sglang",
            pipeline_config=pipeline, performance_mode="manual",
            component_residency=os.environ["SGLANG_RESIDENCY"],
            attention_backend=os.environ["SGLANG_ATTENTION"], warmup_mode="off",
            enable_torch_compile=False, enable_breakable_cuda_graph=False,
            host="127.0.0.1",
        )
        report.update(sglang=importlib.metadata.version("sglang"),
                      sglang_revision=os.environ["SGLANG_REVISION"],
                      offload=os.environ["SGLANG_RESIDENCY"],
                      server_args=dataclasses.asdict(self.generator.server_args))

    def generate(self, config, references, report):
        folder = Path(config["output"])
        result = self.generator.generate(sampling_params_kwargs=dict(
            prompt=config["prompt"], width=int(config["width"]), height=int(config["height"]),
            num_inference_steps=int(config["steps"]), guidance_scale=1.0, true_cfg_scale=1.0,
            seed=int(config["seed"]), num_outputs_per_prompt=1,
            image_path=[str(folder / row["file"]) for row in report["references"]] or None,
            save_output=True, output_path=str(folder), output_file_name="image.png",
        ))
        if isinstance(result, list):
            if len(result) != 1:
                raise RuntimeError("Expected one image from SGLang")
            result = result[0]
        if result is None or not (folder / "image.png").is_file():
            raise RuntimeError("SGLang did not produce an image; see the worker log")
        report["engine_metrics"] = result.metrics
        report["peak_cuda_allocated_gib"] = result.peak_memory_mb / 1024
        peak = result.metrics.get("memory_snapshots", {}).get("after_forward", {}).get("peak_reserved_mb")
        report["peak_cuda_reserved_gib"] = peak / 1024 if peak is not None else None
        # Native SGLang scales editing references to the requested output area.
        report["reference_pixel_budget"] = int(config["width"]) * int(config["height"])
        with Image.open(folder / "image.png") as image:
            return image.copy()

    def close(self):
        self.generator.shutdown()

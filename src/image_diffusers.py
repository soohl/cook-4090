"""Diffusers adapter with component CPU offload and optional W8A8 linears."""

import importlib.metadata
import json
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline


class DiffusersEngine:
    def __init__(self, config, report):
        self.attention = config.get("attention", "sdpa")
        if self.attention not in {"sdpa", "flex-compiled", "comfy-kitchen"}:
            raise ValueError("Unknown IMAGE_ATTENTION setting")
        self.quantization = config.get("quantization", "bf16")
        if self.quantization not in {"bf16", "int8-convrot"}:
            raise ValueError("Unknown IMAGE_QUANTIZATION setting")
        compile_setting = config.get("compile", "off")
        if compile_setting not in {"on", "off"}:
            raise ValueError("IMAGE_COMPILE must be on or off")
        self.compiled = compile_setting == "on" or self.attention == "flex-compiled"
        self.ff_chunk_size = int(config.get("ff_chunk_size", "0"))
        if self.ff_chunk_size < 0:
            raise ValueError("IMAGE_FF_CHUNK_SIZE must be nonnegative")
        self.pipe = QwenImage21Pipeline.from_pretrained(
            config["weights"], dtype=torch.bfloat16, local_files_only=True)
        self.quantization_report = {"quantization": self.quantization}
        if self.quantization == "int8-convrot":
            from .image_quantization import quantize_transformer
            self.quantization_report = quantize_transformer(self.pipe.transformer)
        if self.quantization == "int8-convrot" or self.attention == "comfy-kitchen":
            self.quantization_report["comfy_kitchen"] = importlib.metadata.version("comfy-kitchen")
        if self.ff_chunk_size:
            from .image_memory import ChunkedFeedForward
            for block in self.pipe.transformer.transformer_blocks:
                block.img_mlp = ChunkedFeedForward(block.img_mlp, self.ff_chunk_size)
        self.quantized_attention = None
        if self.attention == "comfy-kitchen":
            from .image_attention import QuantizedAttention
            self.quantized_attention = QuantizedAttention()
            self.pipe.transformer.set_attn_processor(self.quantized_attention)
        if self.attention == "flex-compiled":
            from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor
            self.pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
        if self.compiled:
            self.pipe.transformer.compile_repeated_blocks()
        self.report = report

        def record_tokens(module, args, kwargs):
            self.report["encoder_input_tokens"] = int(kwargs["attention_mask"].sum())
            self.report["conditioning_tokens"] = self.report["encoder_input_tokens"] - self.pipe._drop_idx

        self.pipe.text_encoder.register_forward_pre_hook(record_tokens, with_kwargs=True)
        self.pipe.enable_model_cpu_offload()

    def generate(self, config, references, report):
        self.report = report
        report.update(offload="model_cpu", attention=self.attention)
        report.update(self.quantization_report)
        report["dtype"] = "int8/bfloat16" if self.quantization == "int8-convrot" else "bfloat16"
        report["compile"] = "repeated_blocks" if self.compiled else "none"
        report["ff_chunk_size"] = self.ff_chunk_size
        if report["vae_tiling"]:
            self.pipe.vae.enable_tiling(
                tile_sample_min_height=int(config["vae_tile_size"]),
                tile_sample_min_width=int(config["vae_tile_size"]),
                tile_sample_stride_height=int(config["vae_tile_stride"]),
                tile_sample_stride_width=int(config["vae_tile_stride"]),
            )
        else:
            self.pipe.vae.disable_tiling()
        if self.quantized_attention is not None:
            self.quantized_attention.calls = self.quantized_attention.fallbacks = 0
        progress = Path(config["output"]) / "progress.json"

        def record_progress(pipe, step, timestep, tensors):
            temporary = progress.with_suffix(".tmp")
            temporary.write_text(json.dumps({"completed": step + 1, "total": int(config["steps"])}))
            temporary.replace(progress)
            return tensors

        try:
            image = self.pipe(
                prompt=config["prompt"], width=int(config["width"]), height=int(config["height"]),
                image=references or None, output_resolution=int(config["reference_resolution"]),
                num_inference_steps=int(config["steps"]), true_cfg_scale=1.0, use_kv_cache=True,
                generator=torch.Generator("cuda").manual_seed(int(config["seed"])),
                callback_on_step_end=record_progress, callback_on_step_end_tensor_inputs=[],
            ).images[0]
        finally:
            if self.quantized_attention is not None and not self.compiled:
                report["decode_attention_calls"] = self.quantized_attention.calls
                report["sdpa_prefill_calls"] = self.quantized_attention.fallbacks
        return image

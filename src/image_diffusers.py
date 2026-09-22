"""BF16 Diffusers adapter with component CPU offload."""

import torch
from diffusers import QwenImage21Pipeline


class DiffusersEngine:
    def __init__(self, config, report):
        self.pipe = QwenImage21Pipeline.from_pretrained(
            config["weights"], torch_dtype=torch.bfloat16, local_files_only=True)
        report["offload"] = "model_cpu"

        def record_tokens(module, args, kwargs):
            report["encoder_input_tokens"] = int(kwargs["attention_mask"].sum())
            report["conditioning_tokens"] = report["encoder_input_tokens"] - self.pipe._drop_idx

        self.pipe.text_encoder.register_forward_pre_hook(record_tokens, with_kwargs=True)
        if report["vae_tiling"]:
            self.pipe.vae.enable_tiling(
                tile_sample_min_height=int(config["vae_tile_size"]),
                tile_sample_min_width=int(config["vae_tile_size"]),
                tile_sample_stride_height=int(config["vae_tile_stride"]),
                tile_sample_stride_width=int(config["vae_tile_stride"]),
            )
        self.pipe.enable_model_cpu_offload()

    def generate(self, config, references, report):
        return self.pipe(
            prompt=config["prompt"], width=int(config["width"]), height=int(config["height"]),
            image=references or None, output_resolution=int(config["reference_resolution"]),
            num_inference_steps=int(config["steps"]), true_cfg_scale=1.0, use_kv_cache=True,
            generator=torch.Generator("cuda").manual_seed(int(config["seed"])),
        ).images[0]

    def close(self):
        pass  # The owning worker exits after the request.

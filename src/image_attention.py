"""Optional quantized decode attention; retain Qwen's masked prefill path.

Projection and cache handling follow Hugging Face Diffusers' QwenImage21AttnProcessor
(Apache-2.0). Kernel packages retain their own licenses.
"""

import torch

from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor, _qwenimage21_prepare_qkv,
)


@torch.library.custom_op("cook4090::quantized_decode", mutates_args=())
def quantized_decode(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     mask: torch.Tensor | None) -> torch.Tensor:
    # Keep the data-dependent mask check inside the op, outside the compiled
    # graph. Never discard padding. Qwen's single-image batches are unpadded.
    if mask is not None and bool(mask.all()):
        mask = None
    from comfy_kitchen import int8_attention
    return int8_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                          attn_mask=mask).transpose(1, 2).contiguous()


@quantized_decode.register_fake
def _fake_decode(query, key, value, mask):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


class QuantizedAttention(QwenImage21AttnProcessor):
    def __init__(self):
        self.calls = self.fallbacks = 0

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                 segments=None, key_valid=None):
        # Retain upstream SDPA for block-causal prefill.
        if segments is not None:
            if not torch.compiler.is_compiling():
                self.fallbacks += 1
            return super().__call__(attn, hidden_states, attention_mask, rotary_emb,
                                    layer_cache, kv_cache_mode, cache_write_slice, segments, key_valid)
        query, key, value, length = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice)
        output = quantized_decode(query, key, value, attention_mask)
        if not torch.compiler.is_compiling():
            self.calls += 1
        output = output[:, :length].flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))

"""Quantized decode preserves masks and the model's head layout."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from src.image_attention import QuantizedAttention, QwenImage21AttnProcessor


class AttentionTests(unittest.TestCase):
    def test_prefill_retains_upstream_block_causal_attention(self):
        processor = QuantizedAttention()
        with patch.object(QwenImage21AttnProcessor, "__call__", return_value="masked") as exact:
            segments = [(0, 2, True)]
            self.assertEqual(processor(None, None, segments=segments), "masked")
            self.assertIs(exact.call_args.args[-2], segments)
        self.assertEqual(processor.fallbacks, 1)

    def test_head_layout_and_comfy_padding(self):
        query = torch.arange(24).reshape(1, 3, 2, 4).float()
        masks = (torch.ones(1, 1, 1, 3, dtype=torch.bool), torch.tensor([[[[True, True, False]]]]))
        attn = SimpleNamespace(to_out=[torch.nn.Identity(), torch.nn.Identity()])
        for mask in masks:
            with self.subTest(padded=not bool(mask.all())):
                kernel = Mock(side_effect=lambda q, k, v, **kwargs: q.clone())
                processor = QuantizedAttention()
                with patch.dict(sys.modules, comfy_kitchen=SimpleNamespace(int8_attention=kernel)), patch(
                    "src.image_attention._qwenimage21_prepare_qkv", return_value=(query, query, query, 3)
                ):
                    result = processor(attn, query, attention_mask=mask)
                torch.testing.assert_close(result, query.flatten(2, 3))
                self.assertEqual(processor.calls, 1)
                self.assertEqual(kernel.call_args.args[0].shape, (1, 2, 3, 4))
                passed = kernel.call_args.kwargs["attn_mask"]
                self.assertIsNone(passed) if bool(mask.all()) else self.assertIs(passed, mask)

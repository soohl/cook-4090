"""Feed-forward chunks preserve token order, batches, and partial final chunks."""

import unittest

import torch
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21SwiGLUFeedForward

from src.image_memory import ChunkedFeedForward


class FeedForwardTests(unittest.TestCase):
    def test_chunked_matches_full_feed_forward(self):
        torch.manual_seed(42)
        module = QwenImage21SwiGLUFeedForward(16, 48).eval()
        chunked = ChunkedFeedForward(module, 4)
        for tokens in (3, 4, 8, 11):
            with self.subTest(tokens=tokens), torch.inference_mode():
                inputs = torch.randn(2, tokens, 16)
                torch.testing.assert_close(chunked(inputs), module(inputs))

    def test_rejects_nonpositive_chunk_size(self):
        for size in (0, -1):
            with self.assertRaises(ValueError):
                ChunkedFeedForward(torch.nn.Identity(), size)

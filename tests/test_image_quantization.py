"""Opt-in CUDA check for W8A8 accuracy, compilation, and CPU offload."""

import os
import unittest


@unittest.skipUnless(os.environ.get("COOK_GPU_TESTS") == "1", "opt-in GPU test")
class QuantizationTests(unittest.TestCase):
    def test_chunked_int8_feed_forward_compiles_and_survives_offload(self):
        import torch
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21SwiGLUFeedForward
        from src.image_memory import ChunkedFeedForward
        from src.image_quantization import ConvRotLinear

        torch.manual_seed(42)
        module = QwenImage21SwiGLUFeedForward(512, 1536).to(device="cuda", dtype=torch.bfloat16).eval()
        for name in ("proj", "gate_layer", "out"):
            setattr(module, name, ConvRotLinear(getattr(module, name)).cuda())
        inputs = torch.randn(2, 37, 512, device="cuda", dtype=torch.bfloat16)
        chunked = ChunkedFeedForward(module, 16)
        with torch.inference_mode():
            expected = module(inputs)
            actual = chunked(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Compilation fuses BF16 activation math. Compare both paths compiled.
            expected_compiled = torch.compile(module, fullgraph=True)(inputs)
            compiled = torch.compile(chunked, fullgraph=True)
            torch.testing.assert_close(compiled(inputs), expected_compiled, rtol=0, atol=0)
            chunked.cpu().cuda()
            torch.testing.assert_close(compiled(inputs), expected_compiled, rtol=0, atol=0)

    def test_int8_linear_compiles_and_survives_offload(self):
        import torch
        from src.image_quantization import ConvRotLinear

        torch.manual_seed(42)
        linear = torch.nn.Linear(512, 1024, bias=True, dtype=torch.bfloat16).cuda().eval()
        inputs = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16)
        quantized = ConvRotLinear(linear).cuda()
        self.assertEqual(quantized.weight.dtype, torch.int8)
        with torch.inference_mode():
            expected = linear(inputs)
            actual = quantized(inputs)
            relative_rms = ((actual - expected).float().square().mean()
                            / expected.float().square().mean()).sqrt().item()
            self.assertLess(relative_rms, 0.035)
            compiled = torch.compile(quantized, fullgraph=True)
            torch.testing.assert_close(compiled(inputs), actual, rtol=0, atol=0)
            quantized.cpu().cuda()
            torch.testing.assert_close(compiled(inputs), actual, rtol=0, atol=0)

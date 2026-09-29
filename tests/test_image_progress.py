"""Record real pipeline steps without transferring latent tensors."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.image_diffusers import DiffusersEngine


class ImageProgressTests(unittest.TestCase):
    def test_callback_records_steps_for_each_request(self):
        engine = DiffusersEngine.__new__(DiffusersEngine)
        engine.attention, engine.quantization = "sdpa", "bf16"
        engine.compiled, engine.ff_chunk_size = False, 0
        engine.quantization_report, engine.quantized_attention = {}, None
        engine.pipe = Mock()
        with tempfile.TemporaryDirectory() as directory, patch("src.image_diffusers.torch.Generator"):
            for name, steps in (("first", 3), ("second", 2)):
                output = Path(directory) / name
                output.mkdir()

                def generate(**kwargs):
                    self.assertEqual(kwargs["callback_on_step_end_tensor_inputs"], [])
                    tensors = {}
                    for step in range(steps):
                        self.assertIs(kwargs["callback_on_step_end"](engine.pipe, step, None, tensors), tensors)
                        self.assertEqual(json.loads((output / "progress.json").read_text()),
                                         {"completed": step + 1, "total": steps})
                        self.assertFalse((output / "progress.tmp").exists())
                    return SimpleNamespace(images=["result"])

                engine.pipe.side_effect = generate
                config = dict(output=str(output), prompt="test", width="1024", height="1024",
                              steps=str(steps), seed="42", reference_resolution="1024")
                self.assertEqual(engine.generate(config, [], {"vae_tiling": False}), "result")

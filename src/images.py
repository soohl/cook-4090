"""Validate image requests and delegate to the selected inference engine."""

import json
import os
from pathlib import Path
import secrets
import uuid

from PIL import Image, UnidentifiedImageError

from .registry import profile_env


def size_choices(profile):
    """Resolve the selected model's presets, including local profile overrides."""
    sizes = profile_env(profile)["IMAGE_UI_SIZES"].split(",")
    for size in sizes:
        width, height = (int(part) for part in size.split("x"))
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("Image presets must use positive multiples of 32")
    return sizes


def generate(manager, key, prompt, size, steps, seed, references):
    profile = manager.registry[key]
    if profile["kind"] != "image":
        raise ValueError("Choose an image model")
    env = profile_env(profile)
    if not prompt.strip() or len(prompt) > 4000:
        raise ValueError("Enter a prompt of at most 4,000 characters")
    if size not in size_choices(profile):
        raise ValueError("Choose a listed image size")
    if steps != int(steps) or not 1 <= int(steps) <= int(os.environ["IMAGE_UI_MAX_STEPS"]):
        raise ValueError("Invalid step count")
    if seed != int(seed) or not -1 <= int(seed) < 2**32:
        raise ValueError("Seed must be -1 or an integer from 0 to 4294967295")
    references = references or []
    if len(references) > int(os.environ["IMAGE_MAX_REFERENCES"]):
        raise ValueError("Too many reference images")
    for path in references:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
            raise ValueError("A reference file is not a readable image") from None
    seed = secrets.randbelow(2**32) if seed == -1 else int(seed)
    output = manager.runtime / ("image-" + uuid.uuid4().hex)
    width, height = size.split("x")
    env.update(IMAGE_PROMPT=prompt, IMAGE_WIDTH=width, IMAGE_HEIGHT=height, IMAGE_STEPS=str(int(steps)),
               IMAGE_SEED=str(seed), IMAGE_OUTPUT=str(output), IMAGE_REFERENCES=json.dumps([str(p) for p in references]))
    with manager.lock:
        manager.run_image(key, env, output.with_suffix(".log"))
    report = json.loads((output / "report.json").read_text())
    return str(output / "image.png"), f"{size} · {int(steps)} steps · seed {seed} · {report['generation_seconds']:.1f}s", str(output / "report.json")

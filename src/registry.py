"""Explicit local model/engine profiles. No discovery or downloads from a hub."""

import json
import os
from pathlib import Path

from . import ROOT


def load_registry(path):
    profiles = json.loads(Path(path).read_text())
    registry = {}
    for profile in profiles:
        key = profile["id"]
        if key in registry or profile["kind"] not in {"chat", "image"}:
            raise ValueError(f"Invalid/duplicate model profile: {key}")
        engines = {"chat": {"ninfer", "exllamav3"}, "image": {"diffusers"}}
        if profile["engine"] not in engines[profile["kind"]]:
            raise ValueError(f"Unsupported engine: {profile['engine']}")
        registry[key] = profile
    return registry


def profile_env(profile):
    env = os.environ.copy()
    for key, value in profile.get("env", {}).items():
        env[key] = str(value).replace("${ROOT}", str(ROOT))
    return env


def availability(profile):
    env = profile_env(profile)
    missing = []
    for key in profile["required_env"]:
        value = env.get(key)
        if not value or not Path(value).exists():
            missing.append(key)
    if missing:
        return "Missing " + ", ".join(missing)
    if profile.get("engine") == "exllamav3":
        try:
            exl3_shards(env["EXL3_WEIGHTS"])
        except (OSError, ValueError, KeyError) as exc:
            return "Missing complete EXL3 weights: " + str(exc)
    return "Installed"


def exl3_shards(directory):
    """Reject partial downloads before declaring an EXL3 profile installed."""
    root = Path(directory)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"].values())
    if not names:
        raise ValueError("Empty EXL3 shard index")
    paths = [root / name for name in sorted(names)]
    for path in paths:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing EXL3 shard: {path.name}")
    return paths


def choices(registry, kind):
    return [(p["label"], key) for key, p in registry.items() if p["kind"] == kind and availability(p) == "Installed"]

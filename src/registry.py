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
        if profile["engine"] not in {"ninfer", "llamacpp", "diffusers"}:
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
    return "Missing " + ", ".join(missing) if missing else "Installed"


def choices(registry, kind):
    return [(p["label"], key) for key, p in registry.items() if p["kind"] == kind and availability(p) == "Installed"]

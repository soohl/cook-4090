"""Run upstream TabbyAPI with local EXL3 weights and disposable configuration."""

import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tempfile

from .registry import exl3_shards


def prepare(action):
    """Installation is explicit and may use the network; serving stays offline."""
    env = os.environ
    python = env["EXL3_PYTHON"]
    if action == "setup":
        revision = subprocess.check_output(["git", "-C", env["EXL3_SOURCE"], "rev-parse", "HEAD"], text=True).strip()
        if revision != env["EXL3_SOURCE_REVISION"]:
            raise ValueError("Initialize the pinned backends/tabbyapi submodule")
        commands = []
        if not Path(python).is_file():
            commands.append(["uv", "venv", "--managed-python", "--python", env["EXL3_PYTHON_VERSION"], env["EXL3_ENV"]])
        commands.extend([
            ["uv", "pip", "install", "--only-binary", ":all:", "--python", python, "torch==" + env["EXL3_TORCH"], "--index-url", env["EXL3_TORCH_INDEX"]],
            ["uv", "pip", "install", "--only-binary", ":all:", "--python", python, "-e", env["EXL3_SOURCE"], env["EXL3_WHEEL"]],
        ])
    else:
        commands = [[str(Path(env["EXL3_ENV"]) / "bin/hf"), "download", env["EXL3_MODEL"],
                     "--revision", env["EXL3_REVISION"], "--local-dir", env["EXL3_WEIGHTS"],
                     "--include", "model*.safetensors", "model.safetensors.index.json", "config.json",
                     "quantization_config.json", "tokenizer*", "chat_template.jinja", "generation_config.json",
                     "vocab.json", "merges.txt", "LICENSE", "preprocessor_config.json", "video_preprocessor_config.json"]]
    for command in commands:
        if env.get("DRY_RUN") == "1":
            print(shlex.join(command))
        else:
            subprocess.run(command, check=True)
    if action == "setup" and env.get("DRY_RUN") != "1":
        resolved = subprocess.check_output(["uv", "pip", "freeze", "--python", python], text=True)
        (Path(env["EXL3_ENV"]) / "requirements-resolved.txt").write_text(resolved)


def configuration(args):
    weights = Path(args.weights).resolve()
    if args.context < 1024 or args.context % 256 or args.chunk < 1 or args.draft < 1:
        raise ValueError("Use context >= 1024 divisible by 256, and positive chunk/draft sizes")
    exl3_shards(weights)
    return {
        "network": {"host": "127.0.0.1", "port": args.port, "disable_auth": True,
                    "allowed_origins": [], "disable_fetch_requests": True},
        "model": {"model_dir": str(weights.parent), "model_name": weights.name,
                  "backend": "exllamav3", "max_seq_len": args.context,
                  "cache_size": args.context, "cache_mode": args.cache,
                  "chunk_size": args.chunk, "max_batch_size": 1, "vision": False},
        "draft_model": {"draft_mode": args.spec, "draft_num_tokens": args.draft,
                        "dynamic_draft": False},
        "logging": {"log_live_status": False},
    }


def main():
    if sys.argv[1:] in (["setup"], ["download"]):
        return prepare(sys.argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "weights", "runtime", "cache"):
        parser.add_argument("--" + name, required=True)
    for name in ("port", "context", "chunk", "draft"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--spec", choices=("mtp", "disabled"), required=True)
    args = parser.parse_args()
    config = configuration(args)
    source = Path(args.source).resolve()
    runtime = Path(args.runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exllamav3-", dir=runtime) as directory:
        models = Path(directory) / "models"
        models.mkdir()
        (models / config["model"]["model_name"]).symlink_to(Path(args.weights).resolve(), target_is_directory=True)
        config["model"]["model_dir"] = str(models)
        # JSON is valid YAML. No credentials are needed behind the private edge.
        (Path(directory) / "config.yml").write_text(json.dumps(config))
        os.chdir(directory)
        sys.path.insert(0, str(source))
        sys.argv = [str(source / "main.py")]
        runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()

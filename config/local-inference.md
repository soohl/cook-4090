# Local inference and legacy tools

The primary interface is the [Docker stack](stack.md). This page describes
model installation and optional local tools.

## Setup

Use Linux, an RTX 4090, a compatible NVIDIA driver, Git, `uv`, and
`libseccomp.so.2`. Image weights need about 33.1 GB of disk space, in addition
to the environment and caches. Image tests used 60 GiB of host RAM and reached
about 32 GiB of process memory. This is not a minimum host-memory specification.

Image setup uses Python 3.14.7 and PyTorch 2.14 with its CUDA 13.0 runtime.
Use NVIDIA driver 580 or later. Images do not require a host CUDA toolkit.
Setup creates `build/qwen-image-py314-venv`; the coding environment stays separate.
Use prebuilt wheels for image dependencies. Diffusers uses the pinned source checkout.

1. Clone the repository and its engine submodules.

   ```sh
   git clone --recurse-submodules https://github.com/soohl/cook-4090.git
   ```

2. Enter the repository.

   ```sh
   cd cook-4090
   ```

3. Install the pinned UI and image dependencies.

   ```sh
   ./run.sh image-setup
   ```

4. Download the image model.

   ```sh
   ./run.sh image-download
   ```

5. Start cook-4090.

   ```sh
   ./run.sh legacy-ui
   ```

Open `http://127.0.0.1:7860` on the server. The legacy UI is unauthenticated
and binds to loopback by default. Do not expose it through Funnel.
Stop separately launched GPU servers before you use it.

For an existing checkout, run `git submodule update --init backends/diffusers`,
then rerun `./run.sh image-setup`. The service reuses the image worker until
coding needs the GPU or the service stops. Standalone image commands exit
after each image.

## LLM engines

Build NInfer with `./run.sh setup ninfer`.
This command builds with a pinned CUDA development image in Docker. It does not
download weights or require a host CUDA toolkit. Compilation does not use the GPU.
The executable runs on the host and loads bundled libraries from its build directory.
The host NVIDIA driver remains required. NVIDIA Container Toolkit is needed only
for GPU-enabled containers, not for this build or host inference.
See the [NInfer build requirements](../backends/ninfer/CMakeLists.txt),
and the [NInfer model card](../backends/ninfer/model-cards/Qwen3.8-27B-NInfer/README.md).

Supply local weights through `NINFER_WEIGHTS`. Defaults are in `run.sh`.
The artifact contains its multi-token prediction (MTP) head.
Configure profiles in [config/models.json](models.json).
Restart the UI after configuration changes. The Models tab lists missing artifacts.

### ExLlamaV3 comparison

Use the optional TabbyAPI submodule to serve ExLlamaV3 through the same local
OpenAI-compatible interface. The worker runs on the host and stays offline.
The existing Docker edge provides authentication and private coding access.

```sh
git submodule update --init backends/tabbyapi
./run.sh setup exllamav3
./run.sh exl3-download
```

Setup creates a separate uv-managed Python 3.14.7 environment. It installs
ExLlamaV3 1.5.1 and PyTorch 2.13 wheels with CUDA 13.2. Workers select their
environments automatically. This coding setup requires NVIDIA driver 595 or later.
Download fetches a pinned revision of
[Qwen3.8-27B EXL3 SC 5.00bpw H6 V6](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/SC_5.00bpw_H6_V6).
Weights need about 19.6 GB. Installation requires internet access; serving does not.

Stop host inference, then run `./run.sh --benchmark` to compare installed profiles.
The EXL3 defaults are MTP3, Q4 KV cache, one active request, and a 1024-token
prefill chunk. `EXL3_SPEC=disabled` turns speculation off. Defaults and pins
are in `run.sh`. Resolved packages are saved in the ignored environment directory.

ExLlamaV3 is the default for the private coding API. Start it with `./run.sh`.
To select NInfer, start `CODING_PROFILE=qwen38-ninfer ./run.sh`.
For a standalone loopback server,
run `CONTEXT=32768 ./run.sh serve exllamav3`. Model weights and cache formats
differ between engines; speed measurements do not establish equal quality.
The EXL3 server supports OpenAI chat/completions. It does not add NInfer's
Responses or Anthropic Messages routes.

## Commands and references

- `./run.sh serve ninfer`: standalone LLM API at
  `http://127.0.0.1:8080/v1`. Unlike the default UI profiles, these commands enable
  vision and require the corresponding vision artifacts.
- `./run.sh image`: generate an image and retain its files under `results/qwen-image/`.
- Run `./run.sh --benchmark` for matched terminal comparisons and saved results.
- `./run.sh test`: run CPU-only application checks.
- `./run.sh help`: list commands and configuration options.
- [Ada implementation and measurements](../backends/ninfer/docs/ada.md).

## Layout and model storage

- `src/`: UI, inference adapters, and terminal benchmarking.
- `config/models.json`: local model profiles.
- `backends/`: pinned engine source submodules.
- `models/`: downloaded weights. Qwen-Image-2.1 uses `models/qwen-image-2.1/`.
  The default LLM path is `models/qwen3.8-27b-exl3-5bpw/`.
  NInfer uses `models/dflash2/qwen3_8_27b.ninfer`.
- `assets/`: shared README and UI artwork.
- `tests/`: CPU and browser checks.
- `build/`: environments, caches, and disposable UI sessions.

Model weights, builds, CLI image results, and local `docs/` notes are ignored by Git.
The repository includes an empty `models/` directory for local weights.
The repository uses the [Apache 2.0 license](../LICENSE). Engines and models retain
their own licenses. Qwen-Image-2.1 uses the Qwen Research License.

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Docker edge and host inference. Port 8000 is private; Funnel carries images.
CODING_BIND=${CODING_BIND:-127.0.0.1}
CODING_PROFILE=${CODING_PROFILE:-qwen38-exllamav3}
INFERENCE_SOCKET="$ROOT/build/stack-ipc/inference.sock"
STACK_MAX_BODY=${STACK_MAX_BODY:-33554432}
STACK_LOG_MAX_SIZE=${STACK_LOG_MAX_SIZE:-10m}
STACK_LOG_MAX_FILES=${STACK_LOG_MAX_FILES:-3}
IMAGE_WEB_TIMEOUT=${IMAGE_WEB_TIMEOUT:-7200} # includes queued GPU work
IMAGE_WEB_PORT=${IMAGE_WEB_PORT:-7860}
# Includes web assets, API requests, and the auth proxy's idle connections.
# GPU work has a separate bounded queue (UI_QUEUE).
IMAGE_WEB_CONNECTIONS=${IMAGE_WEB_CONNECTIONS:-256}
INFERENCE_CONNECTIONS=${INFERENCE_CONNECTIONS:-32}
# Caddy listeners: private coding 8000; image TLS 8443; identity TLS 9443.
# Tailscale forwards public 443/8443 and private 8000 to those listeners.
TAILSCALE_IMAGE=${TAILSCALE_IMAGE:-tailscale/tailscale@sha256:2667499ed87ae29218f292556ba062918402dd5e92e93637af14867e4df12dd3}
CADDY_IMAGE=${CADDY_IMAGE:-caddy:2.11.4-alpine@sha256:de23def33b17fb5d1290b0f6c2add1d70780e52341896c00a4c8a2a2fe9d355e}
POCKET_ID_IMAGE=${POCKET_ID_IMAGE:-ghcr.io/pocket-id/pocket-id:v2.16.0@sha256:9366436f3fd21619ed7e5709fa0acac88130f73414ec8ee1caf768fc487111ea}
OAUTH2_PROXY_IMAGE=${OAUTH2_PROXY_IMAGE:-quay.io/oauth2-proxy/oauth2-proxy:v7.15.4@sha256:b1b2021fe8f4004573e8d690dec6c7bb29cc44364572cf8510a05bf3a0ae2ded}
PYTHON_IMAGE=${PYTHON_IMAGE:-python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3}
export CODING_BIND CODING_PROFILE INFERENCE_SOCKET STACK_MAX_BODY IMAGE_WEB_TIMEOUT
export STACK_LOG_MAX_SIZE STACK_LOG_MAX_FILES
export IMAGE_WEB_PORT IMAGE_WEB_CONNECTIONS INFERENCE_CONNECTIONS
NODE_IMAGE=${NODE_IMAGE:-node:22.23.2-alpine@sha256:b6f26b36c8ff49624cfdac716b8ea1138d606df02586a77d364bb5536a634f85}
export TAILSCALE_IMAGE CADDY_IMAGE POCKET_ID_IMAGE OAUTH2_PROXY_IMAGE PYTHON_IMAGE NODE_IMAGE

# Unified cook-4090 UI. Runtime is disposable; weights/builds remain reusable.
UI_HOST=${UI_HOST:-127.0.0.1}
UI_PORT=${UI_PORT:-7860}
UI_ENGINE_PORT=${UI_ENGINE_PORT:-0} # allocate a fresh loopback port per engine start
UI_RUNTIME=${UI_RUNTIME:-"$ROOT/build/ui-runtime"}
UI_REGISTRY=${UI_REGISTRY:-"$ROOT/config/models.json"}
UI_CONTEXT=${UI_CONTEXT:-32768}
UI_MAX_TOKENS=${UI_MAX_TOKENS:-2048}
UI_TEMPERATURE=${UI_TEMPERATURE:-0.7}
UI_SEED=${UI_SEED:-42}
UI_THINKING=${UI_THINKING:-on}
UI_START_TIMEOUT=${UI_START_TIMEOUT:-180}
UI_REQUEST_TIMEOUT=${UI_REQUEST_TIMEOUT:-600}
UI_QUEUE=${UI_QUEUE:-8}
export UI_QUEUE
# Terminal benchmark workloads and conservative verification planning budgets.
VERIFY_RAM_GIB=${VERIFY_RAM_GIB:-48}
VERIFY_VRAM_GIB=${VERIFY_VRAM_GIB:-22}
VERIFY_DISK_GIB=${VERIFY_DISK_GIB:-10}
BENCH_IMAGE_SIZE=${BENCH_IMAGE_SIZE:-1024x1024}
BENCH_IMAGE_STEPS=${BENCH_IMAGE_STEPS:-4}
BENCH_IMAGE_PROMPT=${BENCH_IMAGE_PROMPT:-'A red ceramic teapot on a wooden table, soft window light.'}
export VERIFY_RAM_GIB VERIFY_VRAM_GIB VERIFY_DISK_GIB BENCH_IMAGE_SIZE BENCH_IMAGE_STEPS BENCH_IMAGE_PROMPT
BENCH_TOKENS=${BENCH_TOKENS:-256}
BENCH_REPEATS=${BENCH_REPEATS:-2}
BENCH_SEED=${BENCH_SEED:-42}
BENCH_CACHE=${BENCH_CACHE:-cold}
BENCH_PROMPT=${BENCH_PROMPT:-'Explain how a hash table works, including collisions and resizing, with a short Python example.'}

# Qwen-Image-2.1: W8A8 transformer, BF16 encoder/VAE, component CPU offload.
IMAGE_ENGINE=diffusers
IMAGE_DIFFUSERS_SOURCE=${IMAGE_DIFFUSERS_SOURCE:-"$ROOT/backends/diffusers"}
IMAGE_PYTHON=3.14.7
IMAGE_ENV=${IMAGE_ENV:-"$ROOT/build/qwen-image-py314-venv"}
IMAGE_CACHE=${IMAGE_CACHE:-"$ROOT/build/qwen-image-cache"}
IMAGE_ATTENTION=${IMAGE_ATTENTION:-comfy-kitchen} # comfy-kitchen, sdpa, or flex-compiled
IMAGE_QUANTIZATION=${IMAGE_QUANTIZATION:-int8-convrot} # transformer W8A8; bf16 disables it
IMAGE_COMPILE=${IMAGE_COMPILE:-on} # compile repeated transformer blocks
IMAGE_FF_CHUNK_SIZE=${IMAGE_FF_CHUNK_SIZE:-4096} # tokens per feed-forward pass; 0 disables chunking
IMAGE_REUSE=${IMAGE_REUSE:-on} # retain the pipeline until coding needs the GPU
export IMAGE_CACHE IMAGE_ATTENTION IMAGE_QUANTIZATION IMAGE_COMPILE IMAGE_FF_CHUNK_SIZE IMAGE_REUSE
export TORCHINDUCTOR_CACHE_DIR="$IMAGE_CACHE/inductor" TRITON_CACHE_DIR="$IMAGE_CACHE/triton"
IMAGE_WEIGHTS=${IMAGE_WEIGHTS:-"$ROOT/models/qwen-image-2.1"}
IMAGE_MODEL=Qwen/Qwen-Image-2.1
IMAGE_REVISION=b3179ad355be050328e483a9dfdd9e60cd62adfa
IMAGE_DIFFUSERS_REVISION=4295ee3ec58efa6577bc459e9b84ca3f63aa9a96
IMAGE_TORCH=2.14.0+cu130
IMAGE_TORCHVISION=0.29.0+cu130
IMAGE_TORCH_INDEX=https://download.pytorch.org/whl/cu130
IMAGE_TRANSFORMERS=5.17.0
IMAGE_ACCELERATE=1.15.0
IMAGE_PILLOW=12.3.0
IMAGE_COMFY_KITCHEN=0.2.35
IMAGE_GRADIO=6.28.0
export IMAGE_GRADIO
# Qwen's seven native 2K presets, plus smaller sizes for quick experiments.
IMAGE_UI_SIZES=${IMAGE_UI_SIZES:-1024x1024,2048x2048,2400x1792,1792x2400,2528x1696,1696x2528,2752x1536,1536x2752,1024x768,768x1024,768x768}
IMAGE_VAE_TILING=${IMAGE_VAE_TILING:-auto} # auto above 1 megapixel; on/off overrides
IMAGE_VAE_TILE_SIZE=${IMAGE_VAE_TILE_SIZE:-512}
IMAGE_VAE_TILE_STRIDE=${IMAGE_VAE_TILE_STRIDE:-384}
IMAGE_UI_MAX_STEPS=${IMAGE_UI_MAX_STEPS:-50}
IMAGE_UI_TIMEOUT=${IMAGE_UI_TIMEOUT:-600}
IMAGE_MAX_REFERENCES=${IMAGE_MAX_REFERENCES:-10}
IMAGE_REFERENCES=${IMAGE_REFERENCES:-'[]'} # JSON array of reference-image paths
IMAGE_REFERENCE_RESOLUTION=${IMAGE_REFERENCE_RESOLUTION:-1024}
IMAGE_WIDTH=${IMAGE_WIDTH:-1024}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-1024}
IMAGE_STEPS=${IMAGE_STEPS:-25}
IMAGE_SEED=${IMAGE_SEED:-42}
IMAGE_PROMPT=${IMAGE_PROMPT:-'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement'}
IMAGE_OUTPUT=${IMAGE_OUTPUT:-"$ROOT/results/qwen-image/$(date -u +%Y%m%dT%H%M%SZ)"}

# Best known RTX 4090 configuration. Edit these values for an experiment.
MODEL_ID=${MODEL_ID:-qwen3.8-27b}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CONTEXT=${CONTEXT:-262144}
VISION=${VISION:-on}

NINFER_SOURCE=${NINFER_SOURCE:-"$ROOT/backends/ninfer"}
NINFER_BUILD=${NINFER_BUILD:-"$ROOT/build/ninfer-container-sm89"}
NINFER_GENERATOR=${NINFER_GENERATOR:-"Unix Makefiles"}
NINFER_CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
NINFER_BUILD_IMAGE=cook-4090-ninfer-build:cu128
NINFER_SERVER=${NINFER_SERVER:-"$NINFER_BUILD/apps/ninfer-serve"}
NINFER_WEIGHTS=${NINFER_WEIGHTS:-"$ROOT/models/dflash2/qwen3_8_27b.ninfer"} # bundled MTP, DFlash2 and vision
NINFER_KV=${NINFER_KV:-e8}
NINFER_CHUNK=${NINFER_CHUNK:-1024}
NINFER_SPEC=${NINFER_SPEC:-mtp}
# DFlash2: K5 won the tested code fixture; K7 won structured JSONL. MTP3 remains default.
NINFER_DRAFT=${NINFER_DRAFT:-$([[ "$NINFER_SPEC" == dflash2 ]] && echo 7 || echo 3)}
NINFER_MTP_ADAPTIVE=${NINFER_MTP_ADAPTIVE:-off} # experimental, greedy C1, depth 3..NINFER_DRAFT
NINFER_VISION_TOKENS=${NINFER_VISION_TOKENS:-8192}
NINFER_VISION_OFFLOAD=${NINFER_VISION_OFFLOAD:-off} # on: BF16 vision blocks in pinned RAM, GPU compute
NINFER_PREFIX_REUSE=${NINFER_PREFIX_REUSE:-on}
HOST_STATE_SLOTS=${HOST_STATE_SLOTS:-2}

# Optional EXL3 comparison. The public API still uses the Docker edge.
EXL3_SOURCE=${EXL3_SOURCE:-"$ROOT/backends/tabbyapi"}
EXL3_SOURCE_REVISION=f07131cd8fe34e449fe87cdd3a066b52b96d3cac
EXL3_PYTHON_VERSION=3.14.7
EXL3_ENV=${EXL3_ENV:-"$ROOT/build/exllamav3-py314-venv"}
EXL3_PYTHON=${EXL3_PYTHON:-"$EXL3_ENV/bin/python"}
EXL3_VERSION=1.5.1
EXL3_TORCH=2.13.0+cu132
EXL3_TORCH_INDEX=https://download.pytorch.org/whl/cu132
EXL3_WHEEL="https://github.com/turboderp-org/exllamav3/releases/download/v$EXL3_VERSION/exllamav3-$EXL3_VERSION%2Bcu132.torch2.13.0-cp314-cp314-linux_x86_64.whl"
EXL3_WEIGHTS=${EXL3_WEIGHTS:-"$ROOT/models/qwen3.8-27b-exl3-5bpw"}
EXL3_MODEL=turboderp/Qwen3.8-27B-exl3
EXL3_REVISION=f33f26d929e2b20ef21361145d582f5239e3831f # SC_5.00bpw_H6_V6
EXL3_KV=${EXL3_KV:-Q4}
EXL3_CHUNK=${EXL3_CHUNK:-1024}
EXL3_SPEC=${EXL3_SPEC:-mtp}
EXL3_DRAFT=${EXL3_DRAFT:-3}
EXL3_RUNTIME=${EXL3_RUNTIME:-"$ROOT/build/exllamav3-runtime"}
export EXL3_SOURCE EXL3_PYTHON EXL3_WEIGHTS
export EXL3_SOURCE_REVISION EXL3_ENV EXL3_PYTHON_VERSION EXL3_TORCH EXL3_TORCH_INDEX EXL3_WHEEL EXL3_MODEL EXL3_REVISION


fail() { printf '%s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Missing tool: $1"; }
quote() { printf '%q ' "$@"; printf '\n'; }

check_image_source() {
    local source=$1 revision=$2
    need git
    [[ -e "$source/.git" ]] || fail "Missing engine source: $source. Run git submodule update --init backends/diffusers."
    [[ $(git -C "$source" rev-parse HEAD) == "$revision" ]] || fail "Engine revision mismatch: $source (expected $revision)."
}

serve() {
    local backend=${1:-}
    shift || true
    local -a command files

    case "$backend" in
        ninfer)
            command=(
                "$NINFER_SERVER" "$NINFER_WEIGHTS"
                --model-id "$MODEL_ID" --host "$HOST" --port "$PORT"
                --max-context "$CONTEXT" --kv-capacity "$CONTEXT"
                --prefill-chunk "$NINFER_CHUNK" --kv-dtype "$NINFER_KV"
                --max-concurrency 1 --max-pending-requests 16
                --pending-timeout-ms 600000 --default-max-tokens 16384
                --preserve-thinking
            )
            files=("$NINFER_SERVER" "$NINFER_WEIGHTS")
            case "$NINFER_PREFIX_REUSE" in
                on) command+=(--device-state-slots 0 --host-state-slots "$HOST_STATE_SLOTS" --host-kv-mib 0) ;;
                off) command+=(--no-prefix-reuse) ;;
                *) fail "NINFER_PREFIX_REUSE must be on or off." ;;
            esac
            case "$NINFER_SPEC" in
                mtp|dflash2) command+=(--spec "$NINFER_SPEC" --draft-tokens "$NINFER_DRAFT" --lm-head-draft) ;;
                off) ;;
                *) fail "NINFER_SPEC must be mtp, dflash2, or off." ;;
            esac
            [[ "$VISION" == off ]] || command+=(--vision --vision-max-tokens "$NINFER_VISION_TOKENS")
            case "$NINFER_VISION_OFFLOAD" in
                on)
                    [[ "$VISION" != off ]] || fail "Vision offload requires VISION=on."
                    command+=(--vision-cpu-offload)
                    ;;
                off) ;;
                *) fail "NINFER_VISION_OFFLOAD must be on or off." ;;
            esac
            case "$NINFER_MTP_ADAPTIVE" in
                on) command+=(--mtp-adaptive) ;;
                off) ;;
                *) fail "NINFER_MTP_ADAPTIVE must be on or off." ;;
            esac
            ;;
        exllamav3)
            [[ "$HOST" == 127.0.0.1 ]] || fail "ExLlamaV3 must bind to 127.0.0.1; use the authenticated Docker edge."
            command=("$EXL3_PYTHON" -m src.offline "$EXL3_PYTHON" -m src.exllamav3_server
                --source "$EXL3_SOURCE" --weights "$EXL3_WEIGHTS"
                --runtime "${TMPDIR:-$EXL3_RUNTIME}" --port "$PORT" --context "$CONTEXT"
                --cache "$EXL3_KV" --chunk "$EXL3_CHUNK" --spec "$EXL3_SPEC" --draft "$EXL3_DRAFT")
            files=("$EXL3_PYTHON" "$EXL3_SOURCE/main.py" "$EXL3_WEIGHTS/config.json" "$EXL3_WEIGHTS/model.safetensors.index.json")
            ;;
        *) fail "Supported coding engines: ninfer, exllamav3." ;;
    esac

    command+=("$@")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        quote "${command[@]}"
        return
    fi
    for file in "${files[@]}"; do [[ -e "$file" ]] || fail "Missing: $file"; done
    exec "${command[@]}"
}

setup() {
    local backend=${1:-}
    shift || true
    local -a configure build container image

    case "$backend" in
        ninfer)
            image=(docker build --build-arg "CUDA_IMAGE=$NINFER_CUDA_IMAGE" -t "$NINFER_BUILD_IMAGE" -)
            container=(docker run --rm --network none --user "$(id -u):$(id -g)"
                --mount "type=bind,src=$NINFER_SOURCE,dst=/source,readonly"
                --mount "type=bind,src=$NINFER_BUILD,dst=/build"
                --mount "type=bind,src=$ROOT/config/ninfer-build.sh,dst=/build.sh,readonly"
                --entrypoint "" "$NINFER_BUILD_IMAGE")
            configure=(
                "${container[@]}" cmake -S /source -B /build -G "$NINFER_GENERATOR"
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13
                -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
                -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_ARCHITECTURES=89
                -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF
            )
            build=("${container[@]}" bash /build.sh)
            ;;
        exllamav3)
            exec "$IMAGE_ENV/bin/python" -m src.exllamav3_server setup
            ;;
        *) fail "Supported coding engines: ninfer, exllamav3." ;;
    esac

    configure+=("$@")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        quote "${image[@]}"
        quote "${configure[@]}"
        quote "${build[@]}"
        return
    fi
    need docker
    [[ -f "$NINFER_SOURCE/CMakeLists.txt" ]] || fail "Missing source: $NINFER_SOURCE"
    mkdir -p "$NINFER_BUILD"
    "${image[@]}" < "$ROOT/config/ninfer.Dockerfile"
    "${configure[@]}"
    exec "${build[@]}"
}

usage() {
    cat <<'EOF'
Usage:
  ./run.sh              Start the app and run inference in this terminal.
  ./run.sh --verify     Check setup and available resources.
  ./run.sh --benchmark  Compare installed models and backends in the terminal.

Default coding: ExLlamaV3 / TabbyAPI. Images: Diffusers.
Docker services start automatically. Ctrl+C stops foreground inference.

Configuration defaults are at the top of run.sh.
Setup and advanced commands: config/local-inference.md
Docker, passkeys, and Tailscale: config/stack.md
Benchmark settings: config/commands.md
EOF
}

action=${1:-inference}
shift || true
case "$action" in
    check)
        "$ROOT/run.sh" test
        exec "$ROOT/run.sh" stack check
        ;;
    stack-init|stack|ui|image-ui)
        [[ -x "$IMAGE_ENV/bin/python" ]] || fail "Run ./run.sh image-setup first."
        if [[ "$action" == stack-init ]]; then
            exec "$IMAGE_ENV/bin/python" -m src.stack init
        fi
        if [[ "$action" == ui || "$action" == image-ui ]]; then
            exec "$IMAGE_ENV/bin/python" -m src.stack up
        fi
        exec "$IMAGE_ENV/bin/python" -m src.stack "$@"
        ;;
    image-setup)
        need uv
        check_image_source "$IMAGE_DIFFUSERS_SOURCE" "$IMAGE_DIFFUSERS_REVISION"
        [[ -x "$IMAGE_ENV/bin/python" ]] || uv venv --managed-python --python "$IMAGE_PYTHON" "$IMAGE_ENV"
        uv pip install --only-binary :all: --python "$IMAGE_ENV/bin/python" "torch==$IMAGE_TORCH" "torchvision==$IMAGE_TORCHVISION" --index-url "$IMAGE_TORCH_INDEX"
        uv pip install --only-binary :all: --python "$IMAGE_ENV/bin/python" \
            -r "$ROOT/config/web-requirements.txt" \
            -e "$IMAGE_DIFFUSERS_SOURCE" \
            "transformers==$IMAGE_TRANSFORMERS" "accelerate==$IMAGE_ACCELERATE" "pillow==$IMAGE_PILLOW" "gradio==$IMAGE_GRADIO" \
            "comfy-kitchen==$IMAGE_COMFY_KITCHEN"
        mkdir -p "$IMAGE_CACHE"
        uv pip freeze --python "$IMAGE_ENV/bin/python" > "$IMAGE_CACHE/requirements-resolved.txt"
        ;;
    exl3-download)
        exec "$IMAGE_ENV/bin/python" -m src.exllamav3_server download
        ;;
    image-download)
        [[ -x "$IMAGE_ENV/bin/hf" ]] || fail "Run ./run.sh image-setup first."
        export HF_HOME="$IMAGE_CACHE/huggingface"
        exec "$IMAGE_ENV/bin/hf" download "$IMAGE_MODEL" --revision "$IMAGE_REVISION" --local-dir "$IMAGE_WEIGHTS"
        ;;
    inference|--verify|--benchmark|image|legacy-ui|ui-test|test)
        if [[ ! -x "$IMAGE_ENV/bin/python" ]]; then
            printf '\n  cook-4090 / setup\n  FAIL   Python environment        Run: ./run.sh image-setup\n  Result: needs attention\n\n' >&2
            exit 1
        fi
        export HF_HOME="$IMAGE_CACHE/huggingface"
        export IMAGE_ENGINE IMAGE_ENV IMAGE_WEIGHTS IMAGE_MODEL IMAGE_REVISION IMAGE_DIFFUSERS_SOURCE IMAGE_DIFFUSERS_REVISION
        export IMAGE_WIDTH IMAGE_HEIGHT IMAGE_STEPS IMAGE_SEED IMAGE_PROMPT IMAGE_OUTPUT
        export IMAGE_REFERENCES IMAGE_MAX_REFERENCES IMAGE_REFERENCE_RESOLUTION
        export IMAGE_VAE_TILING IMAGE_VAE_TILE_SIZE IMAGE_VAE_TILE_STRIDE
        if [[ "$action" == --verify || "$action" == --benchmark || "$action" == inference || "$action" == legacy-ui || "$action" == ui-test || "$action" == test ]]; then
            export UI_HOST UI_PORT UI_ENGINE_PORT UI_RUNTIME UI_REGISTRY UI_CONTEXT
            export UI_MAX_TOKENS UI_TEMPERATURE UI_SEED UI_THINKING UI_START_TIMEOUT UI_REQUEST_TIMEOUT UI_QUEUE
            export BENCH_TOKENS BENCH_REPEATS BENCH_SEED BENCH_CACHE BENCH_PROMPT
            export IMAGE_UI_SIZES IMAGE_UI_MAX_STEPS IMAGE_UI_TIMEOUT
            export NINFER_SERVER NINFER_WEIGHTS
            export MODEL_ID
            cd "$ROOT"
            if [[ "$action" == --verify || "$action" == --benchmark ]]; then
                exec "$IMAGE_ENV/bin/python" -m src.commands "$action"
            fi
            if [[ "$action" == inference ]]; then
                "$IMAGE_ENV/bin/python" -m src.stack up
                exec "$IMAGE_ENV/bin/python" -u -m src.inference_service
            fi
            if [[ "$action" == ui-test || "$action" == test ]]; then
                exec "$IMAGE_ENV/bin/python" -m unittest discover -s tests -p 'test_*.py' -v
            fi
            exec "$IMAGE_ENV/bin/python" -u -m src.app
        fi
        exec "$IMAGE_ENV/bin/python" -m src.offline "$IMAGE_ENV/bin/python" -u -m src.image_worker
        ;;
    setup) setup "$@" ;;
    serve) serve "$@" ;;
    help|-h|--help|"") usage ;;
    *) fail "Unknown action: $action" ;;
esac

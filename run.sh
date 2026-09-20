#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Unified cook-4090 UI. Runtime is disposable; weights/builds remain reusable.
UI_HOST=${UI_HOST:-0.0.0.0}
UI_PORT=${UI_PORT:-7860}
UI_ENGINE_PORT=${UI_ENGINE_PORT:-0} # allocate a fresh loopback port per engine start
UI_RUNTIME=${UI_RUNTIME:-"$ROOT/build/ui-runtime"}
UI_REGISTRY=${UI_REGISTRY:-"$ROOT/config/models.json"}
UI_CONTEXT=${UI_CONTEXT:-32768}
UI_MAX_TOKENS=${UI_MAX_TOKENS:-2048}
UI_TEMPERATURE=${UI_TEMPERATURE:-0.7}
UI_THINKING=${UI_THINKING:-on}
UI_START_TIMEOUT=${UI_START_TIMEOUT:-180}
UI_REQUEST_TIMEOUT=${UI_REQUEST_TIMEOUT:-600}
UI_QUEUE=${UI_QUEUE:-8}
UI_BENCH_TOKENS=${UI_BENCH_TOKENS:-256}
UI_BENCH_REPEATS=${UI_BENCH_REPEATS:-2}
UI_BENCH_SEED=${UI_BENCH_SEED:-42}
UI_BENCH_CACHE=${UI_BENCH_CACHE:-cold}
UI_BENCH_PROMPT=${UI_BENCH_PROMPT:-'Explain how a hash table works, including collisions and resizing, with a short Python example.'}

# Qwen-Image-2.1: isolated BF16 diffusion experiment with component CPU offload.
IMAGE_ENV=${IMAGE_ENV:-"$ROOT/build/qwen-image-venv"}
IMAGE_CACHE=${IMAGE_CACHE:-"$ROOT/build/qwen-image-cache"}
IMAGE_WEIGHTS=${IMAGE_WEIGHTS:-"$ROOT/models/qwen-image-2.1"}
IMAGE_MODEL=Qwen/Qwen-Image-2.1
IMAGE_REVISION=b3179ad355be050328e483a9dfdd9e60cd62adfa
IMAGE_DIFFUSERS_REVISION=80c7ed262aeffbeb43ef13ae04baeb9b84515a69
IMAGE_TORCH=2.8.0+cu128
IMAGE_TORCHVISION=0.23.0+cu128
IMAGE_TORCH_INDEX=https://download.pytorch.org/whl/cu128
IMAGE_TRANSFORMERS=5.17.0
IMAGE_ACCELERATE=1.15.0
IMAGE_PILLOW=12.3.0
IMAGE_GRADIO=6.28.0
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
IMAGE_STEPS=${IMAGE_STEPS:-40}
IMAGE_SEED=${IMAGE_SEED:-42}
IMAGE_PROMPT=${IMAGE_PROMPT:-'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement'}
IMAGE_OUTPUT=${IMAGE_OUTPUT:-"$ROOT/results/qwen-image/$(date -u +%Y%m%dT%H%M%SZ)"}

# Best known RTX 4090 configuration. Edit these values for an experiment.
MODEL_ID=${MODEL_ID:-qwen3.8-27b}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CONTEXT=${CONTEXT:-$([[ "${2:-}" == llamacpp ]] && echo 32768 || echo 262144)}
MTP=${MTP:-inline} # on: external llama.cpp draft; inline: fused GGUF; off: disabled
VISION=${VISION:-on}

CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.8}
NINFER_SOURCE=${NINFER_SOURCE:-"$ROOT/backends/ninfer"}
NINFER_BUILD=${NINFER_BUILD:-"$ROOT/build/ninfer-sm89"}
NINFER_GENERATOR=${NINFER_GENERATOR:-"Unix Makefiles"}
NINFER_SERVER=${NINFER_SERVER:-"$NINFER_BUILD/apps/ninfer-serve"}
NINFER_WEIGHTS=${NINFER_WEIGHTS:-"$ROOT/models/dflash2/qwen3_8_27b.ninfer"} # bundled MTP, DFlash2 and vision
NINFER_KV=${NINFER_KV:-e8}
NINFER_CHUNK=${NINFER_CHUNK:-1024}
NINFER_SPEC=${NINFER_SPEC:-$([[ "$MTP" == off ]] && echo off || echo mtp)}
# DFlash2: K5 won the tested code fixture; K7 won structured JSONL. MTP3 remains default.
NINFER_DRAFT=${NINFER_DRAFT:-$([[ "$NINFER_SPEC" == dflash2 ]] && echo 7 || echo 3)}
NINFER_MTP_ADAPTIVE=${NINFER_MTP_ADAPTIVE:-off} # experimental, greedy C1, depth 3..NINFER_DRAFT
NINFER_VISION_TOKENS=${NINFER_VISION_TOKENS:-8192}
NINFER_VISION_OFFLOAD=${NINFER_VISION_OFFLOAD:-off} # on: BF16 vision blocks in pinned RAM, GPU compute
NINFER_PREFIX_REUSE=${NINFER_PREFIX_REUSE:-on}
HOST_STATE_SLOTS=${HOST_STATE_SLOTS:-2}

LLAMACPP_SOURCE=${LLAMACPP_SOURCE:-"$ROOT/backends/llamacpp"}
LLAMACPP_BUILD=${LLAMACPP_BUILD:-"$ROOT/build/llamacpp-comparison"}
LLAMACPP_GENERATOR=${LLAMACPP_GENERATOR:-"Unix Makefiles"}
LLAMACPP_SERVER=${LLAMACPP_SERVER:-"$LLAMACPP_BUILD/bin/llama-server"}
LLAMACPP_REVISION=${LLAMACPP_REVISION:-7eda61931e46df9a46132833f5734284fa89df92}
GGUF_WEIGHTS=${GGUF_WEIGHTS:-"$ROOT/models/comparison/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf"}
MTP_WEIGHTS=${MTP_WEIGHTS:-"$ROOT/models/mtp-Qwen3.8-27B-Q4_0.gguf"}
VISION_WEIGHTS=${VISION_WEIGHTS:-"$ROOT/models/comparison/unsloth/mmproj-F16.gguf"}
LLAMACPP_KV=${LLAMACPP_KV:-q4_0}
LLAMACPP_BATCH=${LLAMACPP_BATCH:-1024}
LLAMACPP_UBATCH=${LLAMACPP_UBATCH:-256}
LLAMACPP_THREADS=${LLAMACPP_THREADS:-8}
LLAMACPP_DRAFT=${LLAMACPP_DRAFT:-4}
LLAMACPP_UI=${LLAMACPP_UI:-on}
LLAMACPP_UI_GZIP=${LLAMACPP_UI_GZIP:-OFF} # embedded UI works without Accept-Encoding: gzip


fail() { printf '%s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Missing tool: $1"; }
quote() { printf '%q ' "$@"; printf '\n'; }

serve() {
    local backend=${1:-}
    shift || true
    local -a command files

    case "$MTP" in
        on|inline|off) ;;
        *) fail "MTP must be on, inline, or off." ;;
    esac

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
        llamacpp)
            command=(
                "$LLAMACPP_SERVER"
                --model "$GGUF_WEIGHTS" --alias "$MODEL_ID"
                --host "$HOST" --port "$PORT" --device CUDA0 --n-gpu-layers 99
                --ctx-size "$CONTEXT" --parallel 1
                --batch-size "$LLAMACPP_BATCH" --ubatch-size "$LLAMACPP_UBATCH"
                --threads "$LLAMACPP_THREADS" --threads-batch "$LLAMACPP_THREADS"
                --cache-type-k "$LLAMACPP_KV" --cache-type-v "$LLAMACPP_KV"
                --flash-attn on --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
            )
            files=("$LLAMACPP_SERVER" "$GGUF_WEIGHTS")
            case "$LLAMACPP_UI" in
                on) command+=(--ui) ;;
                off) command+=(--no-ui) ;;
                *) fail "LLAMACPP_UI must be on or off." ;;
            esac
            if [[ "$MTP" != off ]]; then
                command+=(
                    --spec-type draft-mtp
                    --spec-draft-device CUDA0 --spec-draft-ngl 99
                    --spec-draft-n-max "$LLAMACPP_DRAFT" --spec-draft-p-min 0.30
                )
                if [[ "$MTP" == on ]]; then
                    command+=(--spec-draft-model "$MTP_WEIGHTS")
                    files+=("$MTP_WEIGHTS")
                fi
            fi
            if [[ "$VISION" != off ]]; then
                command+=(--mmproj "$VISION_WEIGHTS")
                files+=("$VISION_WEIGHTS")
            fi
            ;;
        *) fail "Choose ninfer or llamacpp." ;;
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
    local -a configure build

    case "$backend" in
        ninfer)
            configure=(
                cmake -S "$NINFER_SOURCE" -B "$NINFER_BUILD" -G "$NINFER_GENERATOR"
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13
                "-DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc"
                "-DCUDAToolkit_ROOT=$CUDA_ROOT" -DCMAKE_CUDA_ARCHITECTURES=89
                -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF
            )
            build=(cmake --build "$NINFER_BUILD" -j --target ninfer ninfer-serve)
            ;;
        llamacpp)
            configure=(
                cmake -S "$LLAMACPP_SOURCE" -B "$LLAMACPP_BUILD" -G "$LLAMACPP_GENERATOR"
                -DCMAKE_BUILD_TYPE=Release
                "-DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc"
                "-DCUDAToolkit_ROOT=$CUDA_ROOT" -DCMAKE_CUDA_ARCHITECTURES=89
                -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DGGML_NATIVE=ON
                -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
                -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_EXAMPLES=OFF
                -DLLAMA_USE_PREBUILT_UI=ON "-DLLAMA_UI_GZIP=$LLAMACPP_UI_GZIP"
                -DLLAMA_BUILD_TESTS=OFF
            )
            build=(cmake --build "$LLAMACPP_BUILD" -j --target llama-server)
            ;;
        *) fail "Choose ninfer or llamacpp." ;;
    esac

    configure+=("$@")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        quote "${configure[@]}"
        quote "${build[@]}"
        return
    fi
    need cmake
    [[ -f "${configure[2]}/CMakeLists.txt" ]] || fail "Missing source: ${configure[2]}"
    [[ -x "$CUDA_ROOT/bin/nvcc" ]] || fail "Missing CUDA compiler: $CUDA_ROOT/bin/nvcc"
    if [[ "$backend" == llamacpp ]]; then
        need git
        [[ $(git -C "$LLAMACPP_SOURCE" rev-parse HEAD) == "$LLAMACPP_REVISION" ]] ||
            fail "llama.cpp revision does not match LLAMACPP_REVISION."
    fi
    "${configure[@]}"
    exec "${build[@]}"
}

usage() {
    cat <<'EOF'
Usage:
  ./run.sh ui # offline LAN UI: chat, images, matched LLM benchmarks
  ./run.sh test # lifecycle, offline, streaming and GUI benchmark checks
  ./run.sh ui-test # compatibility alias for ./run.sh test
  ./run.sh image-setup # isolated pinned Diffusers environment
  ./run.sh image-download # pinned official BF16 weights (~33 GB)
  ./run.sh image # 1024x1024, 40 steps, seed 42, model CPU offload
  ./run.sh image-ui # compatibility alias for ./run.sh ui
  IMAGE_PROMPT='A capybara reading a book' ./run.sh image
  ./run.sh setup {ninfer|llamacpp}
  ./run.sh serve {ninfer|llamacpp}

MTP=on uses an external llama.cpp draft; MTP=inline uses the GGUF's MTP head.
NInfer embeds MTP in its artifact for both on and inline. MTP=off disables MTP.
NINFER_SPEC=mtp|dflash2|off overrides the NInfer speculative mode.
DFlash2 defaults to seven drafts and requires companion weights in NINFER_WEIGHTS.
NINFER_DRAFT=5 selects the faster tested DFlash2 code profile; seven favors structured JSONL.
Use CONTEXT=32768 for the tested DFlash2 vision profile.
NINFER_VISION_OFFLOAD=on requires the BF16-vision artifact; compute and MTP stay on GPU.
NINFER_PREFIX_REUSE=off disables prefix reuse for fresh-request measurements.

Edit the configuration block at the top of run.sh or override a value in the
environment for one experiment.
EOF
}

action=${1:-}
shift || true
case "$action" in
    image-setup)
        need uv
        [[ -x "$IMAGE_ENV/bin/python" ]] || uv venv --python 3.11 "$IMAGE_ENV"
        uv pip install --python "$IMAGE_ENV/bin/python" "torch==$IMAGE_TORCH" "torchvision==$IMAGE_TORCHVISION" --index-url "$IMAGE_TORCH_INDEX"
        uv pip install --python "$IMAGE_ENV/bin/python" \
            "diffusers @ git+https://github.com/huggingface/diffusers@$IMAGE_DIFFUSERS_REVISION" \
            "transformers==$IMAGE_TRANSFORMERS" "accelerate==$IMAGE_ACCELERATE" "pillow==$IMAGE_PILLOW" "gradio==$IMAGE_GRADIO"
        mkdir -p "$IMAGE_CACHE"
        uv pip freeze --python "$IMAGE_ENV/bin/python" > "$IMAGE_CACHE/requirements-resolved.txt"
        ;;
    image-download)
        [[ -x "$IMAGE_ENV/bin/hf" ]] || fail "Run ./run.sh image-setup first."
        export HF_HOME="$IMAGE_CACHE/huggingface"
        exec "$IMAGE_ENV/bin/hf" download "$IMAGE_MODEL" --revision "$IMAGE_REVISION" --local-dir "$IMAGE_WEIGHTS"
        ;;
    image|image-ui|ui|ui-test|test)
        [[ -x "$IMAGE_ENV/bin/python" ]] || fail "Run ./run.sh image-setup first."
        export HF_HOME="$IMAGE_CACHE/huggingface"
        export IMAGE_WEIGHTS IMAGE_MODEL IMAGE_REVISION IMAGE_DIFFUSERS_REVISION
        export IMAGE_WIDTH IMAGE_HEIGHT IMAGE_STEPS IMAGE_SEED IMAGE_PROMPT IMAGE_OUTPUT
        export IMAGE_REFERENCES IMAGE_MAX_REFERENCES IMAGE_REFERENCE_RESOLUTION
        export IMAGE_VAE_TILING IMAGE_VAE_TILE_SIZE IMAGE_VAE_TILE_STRIDE
        if [[ "$action" == ui || "$action" == image-ui || "$action" == ui-test || "$action" == test ]]; then
            export UI_HOST UI_PORT UI_ENGINE_PORT UI_RUNTIME UI_REGISTRY UI_CONTEXT
            export UI_MAX_TOKENS UI_TEMPERATURE UI_THINKING UI_START_TIMEOUT UI_REQUEST_TIMEOUT UI_QUEUE
            export UI_BENCH_TOKENS UI_BENCH_REPEATS UI_BENCH_SEED UI_BENCH_CACHE UI_BENCH_PROMPT
            export IMAGE_UI_SIZES IMAGE_UI_MAX_STEPS IMAGE_UI_TIMEOUT
            export NINFER_SERVER NINFER_WEIGHTS GGUF_WEIGHTS LLAMACPP_SERVER
            cd "$ROOT"
            if [[ "$action" == ui-test || "$action" == test ]]; then
                exec "$IMAGE_ENV/bin/python" -m unittest discover -s tests -p 'test_*.py' -v
            fi
            exec "$IMAGE_ENV/bin/python" -u -m src.app
        fi
        exec "$IMAGE_ENV/bin/python" -u -m src.image_worker
        ;;
    setup) setup "$@" ;;
    serve) serve "$@" ;;
    help|-h|--help|"") usage ;;
    *) fail "Unknown action: $action" ;;
esac

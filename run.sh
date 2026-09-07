#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Best known RTX 4090 configuration. Edit these values for an experiment.
MODEL_ID=${MODEL_ID:-qwen3.8-27b}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CONTEXT=${CONTEXT:-262144}
MTP=${MTP:-on}
VISION=${VISION:-on}

CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.8}
NINFER_SOURCE=${NINFER_SOURCE:-"$ROOT/backends/ninfer"}
NINFER_BUILD=${NINFER_BUILD:-"$ROOT/build/ninfer"}
NINFER_SERVER=${NINFER_SERVER:-"$NINFER_BUILD/apps/ninfer-serve"}
NINFER_WEIGHTS=${NINFER_WEIGHTS:-"$ROOT/models/qwen3_8_27b.ninfer"}
NINFER_KV=${NINFER_KV:-e8}
NINFER_CHUNK=${NINFER_CHUNK:-1024}
NINFER_DRAFT=${NINFER_DRAFT:-3}
NINFER_VISION_TOKENS=${NINFER_VISION_TOKENS:-8192}
HOST_STATE_SLOTS=${HOST_STATE_SLOTS:-2}

LLAMACPP_SOURCE=${LLAMACPP_SOURCE:-"$ROOT/backends/llamacpp"}
LLAMACPP_BUILD=${LLAMACPP_BUILD:-"$ROOT/build/llamacpp"}
LLAMACPP_SERVER=${LLAMACPP_SERVER:-"$LLAMACPP_BUILD/bin/llama-server"}
LLAMACPP_REVISION=${LLAMACPP_REVISION:-5266f24da75dc449bd56cbed7addb9c8e4a6a73e}
GGUF_WEIGHTS=${GGUF_WEIGHTS:-"$ROOT/models/Qwen3.8-27B-UD-IQ4_XS.gguf"}
MTP_WEIGHTS=${MTP_WEIGHTS:-"$ROOT/models/mtp-Qwen3.8-27B-Q4_0.gguf"}
VISION_WEIGHTS=${VISION_WEIGHTS:-"$ROOT/models/mmproj-Qwen3.8-27B-Q8_0.gguf"}
LLAMACPP_KV=${LLAMACPP_KV:-q4_0}
LLAMACPP_BATCH=${LLAMACPP_BATCH:-1024}
LLAMACPP_UBATCH=${LLAMACPP_UBATCH:-256}
LLAMACPP_THREADS=${LLAMACPP_THREADS:-8}
LLAMACPP_DRAFT=${LLAMACPP_DRAFT:-4}

BENCHMARK_WARMUP=${BENCHMARK_WARMUP:-0}
BENCHMARK_REPEATS=${BENCHMARK_REPEATS:-1}
BENCHMARK_TOKENS=${BENCHMARK_TOKENS:-256}
BENCHMARK_TIMEOUT=${BENCHMARK_TIMEOUT:-600}

fail() { printf '%s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Missing tool: $1"; }
quote() { printf '%q ' "$@"; printf '\n'; }

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
                --device-state-slots 0 --host-state-slots "$HOST_STATE_SLOTS"
                --host-kv-mib 0 --preserve-thinking
            )
            files=("$NINFER_SERVER" "$NINFER_WEIGHTS")
            [[ "$MTP" == off ]] || command+=(--spec mtp --draft-tokens "$NINFER_DRAFT" --lm-head-draft)
            [[ "$VISION" == off ]] || command+=(--vision --vision-max-tokens "$NINFER_VISION_TOKENS")
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
            if [[ "$MTP" != off ]]; then
                command+=(
                    --spec-type draft-mtp --spec-draft-model "$MTP_WEIGHTS"
                    --spec-draft-device CUDA0 --spec-draft-ngl 99
                    --spec-draft-n-max "$LLAMACPP_DRAFT" --spec-draft-p-min 0.30
                )
                files+=("$MTP_WEIGHTS")
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
                cmake -S "$NINFER_SOURCE" -B "$NINFER_BUILD" -G Ninja
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
                cmake -S "$LLAMACPP_SOURCE" -B "$LLAMACPP_BUILD" -G Ninja
                -DCMAKE_BUILD_TYPE=Release
                "-DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc"
                "-DCUDAToolkit_ROOT=$CUDA_ROOT" -DCMAKE_CUDA_ARCHITECTURES=89
                -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DGGML_NATIVE=ON
                -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF
                -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_EXAMPLES=OFF
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
  ./run.sh setup {ninfer|llamacpp}
  ./run.sh serve {ninfer|llamacpp}
  ./run.sh benchmark

Edit the configuration block at the top of run.sh or override a value in the
environment for one experiment.
EOF
}

action=${1:-}
shift || true
case "$action" in
    setup) setup "$@" ;;
    serve) serve "$@" ;;
    benchmark)
        need uv
        export BENCHMARK_WARMUP BENCHMARK_REPEATS BENCHMARK_TOKENS BENCHMARK_TIMEOUT
        exec "$ROOT/benchmark.py"
        ;;
    help|-h|--help|"") usage ;;
    *) fail "Unknown action: $action" ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Best known RTX 4090 configuration. Edit these values for an experiment.
MODEL_ID=${MODEL_ID:-qwen3.8-27b}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CONTEXT=${CONTEXT:-$([[ "${2:-}" == llamacpp || "${2:-}" == exl3 ]] && echo 32768 || echo 262144)}
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

BENCHMARK_WARMUP=${BENCHMARK_WARMUP:-0}
BENCHMARK_REPEATS=${BENCHMARK_REPEATS:-1}
BENCHMARK_TOKENS=${BENCHMARK_TOKENS:-32}
BENCHMARK_TIMEOUT=${BENCHMARK_TIMEOUT:-600}

# Optional comparison engine; checkout and artifacts are local and ignored.
Q27_SOURCE=${Q27_SOURCE:-"$ROOT/backends/q27"}
Q27_SERVER=${Q27_SERVER:-"$Q27_SOURCE/build/q27-server-w8"}
Q27_WEIGHTS=${Q27_WEIGHTS:-"$ROOT/models/comparison/q27/qwen38-27b-mtp-q4s.q27"}
Q27_TOKENIZER=${Q27_TOKENIZER:-"$ROOT/models/comparison/q27/qwen38-27b-mtp.tok"}
Q27_KV=${Q27_KV:-fp8}
Q27_MAXD=${Q27_MAXD:-auto7}
Q27_PMIN=${Q27_PMIN:-0.5}
Q27_SUFFIX_W=${Q27_SUFFIX_W:-8}
Q27_SUFFIX=${Q27_SUFFIX:-1}
# q27's upstream serving profile: FP8 KV, adaptive MTP + suffix drafting, width 8.
# These differ from NInfer E8 and llama.cpp q4_0; compare deployments, not arithmetic.
CROSS_CONTEXT=${CROSS_CONTEXT:-32768}
CROSS_REPEATS=${CROSS_REPEATS:-2}
CROSS_TOKENS=${CROSS_TOKENS:-1024}
CROSS_WARMUP_TOKENS=${CROSS_WARMUP_TOKENS:-32}
CROSS_MTP_DRAFT=${CROSS_MTP_DRAFT:-3}
CROSS_TIMEOUT=${CROSS_TIMEOUT:-600}
CROSS_THINKING=${CROSS_THINKING:-off}
CROSS_QUALITY_TOKENS=${CROSS_QUALITY_TOKENS:-128}
# Optional MiaAI EXL3 deployment comparison; all downloaded artifacts are ignored.
EXL3_SOURCE=${EXL3_SOURCE:-"$ROOT/backends/exllamav3"}
EXL3_KIT=${EXL3_KIT:-"$ROOT/backends/mia-exl3"}
EXL3_REVISION=${EXL3_REVISION:-63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed}
EXL3_KIT_REVISION=${EXL3_KIT_REVISION:-1242187390f780dba907e30659f1639fd0b491c8}
EXL3_ENV=${EXL3_ENV:-"$ROOT/build/exllamav3-venv"}
EXL3_BUILD=${EXL3_BUILD:-"$ROOT/build/exllamav3-sm89"}
EXL3_CACHE=${EXL3_CACHE:-"$ROOT/build/exllamav3-cache"}
EXL3_WEIGHTS=${EXL3_WEIGHTS:-"$ROOT/models/comparison/mia-exl3/Qwen3.8-27B-EXL3-3.5bpw"}
EXL3_DRAFT_WEIGHTS=${EXL3_DRAFT_WEIGHTS:-"$ROOT/models/comparison/mia-exl3/Qwen3.8-27B-DFlash2-EXL3-5.0bpw"}
EXL3_WEIGHTS_REVISION=${EXL3_WEIGHTS_REVISION:-19441ac874c4018295da848e250f23511361cda4}
EXL3_DRAFT_REVISION=${EXL3_DRAFT_REVISION:-4f0436269bca761b071f05319e8e04a87cc633f9}
EXL3_KV=${EXL3_KV:-nvfp4}
# Pinned upstream server defaults: MTP4, DFlash2 K7, 2,048-token prefill chunks,
# xhigh thinking, one active request, automatic compatible-prefix reuse.
# It does not expose these settings as CLI flags; this comparison preserves them.
EXL3_SPEC=${EXL3_SPEC:-mtp}
EXL3_CONTEXT=${EXL3_CONTEXT:-32768}
EXL3_GPU_GB=${EXL3_GPU_GB:-22}
EXL3_BUILD_JOBS=${EXL3_BUILD_JOBS:-4}
EXL3_TORCH=${EXL3_TORCH:-2.8.0+cu128}
EXL3_TORCH_INDEX=${EXL3_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
EXL3_COMPARE_QUALITY_TOKENS=${EXL3_COMPARE_QUALITY_TOKENS:-8192}
QUALITY_PROFILES=${QUALITY_PROFILES:-bf16:32768,int8:32768,e8:32768,bf16:65536,e8:65536,int8:131072,e8:131072,e8:262144}
QUALITY_OUTPUT=${QUALITY_OUTPUT:-8192}
QUALITY_SEED=${QUALITY_SEED:-42}
QUALITY_TIMEOUT=${QUALITY_TIMEOUT:-1800}
QUALITY_SPEC=${QUALITY_SPEC:-mtp} # fixed MTP3 across KV modes; off is a separate control
QUALITY_THINKING_BUDGET=${QUALITY_THINKING_BUDGET:-6144} # reserve room for final answer within 8192
QUALITY_INPUT_TOKENS=${QUALITY_INPUT_TOKENS:-} # optional approximate input override; 0 removes filler
VISION_PYTHON=${VISION_PYTHON:-"$ROOT/.venv-vision/bin/python"}
VISION_BF16_SOURCE=${VISION_BF16_SOURCE:-"$ROOT/models/vision-bf16-source"}
VISION_BF16_WEIGHTS=${VISION_BF16_WEIGHTS:-"$ROOT/models/vision-bf16/qwen3_8_27b.ninfer"}
# Higher-precision KV exploration; serving defaults above are not changed.
KV_CONTEXT_MODES=${KV_CONTEXT_MODES:-fp8,int8}
KV_CONTEXT_START=${KV_CONTEXT_START:-131072}
KV_CONTEXT_STEP=${KV_CONTEXT_STEP:-8192}
KV_CONTEXT_MAX=${KV_CONTEXT_MAX:-196608}
KV_CONTEXT_HEADROOM_MIB=${KV_CONTEXT_HEADROOM_MIB:-256}
KV_CONTEXT_OUTPUT=${KV_CONTEXT_OUTPUT:-16384}
KV_CONTEXT_SEED=${KV_CONTEXT_SEED:-42}
KV_CONTEXT_TIMEOUT=${KV_CONTEXT_TIMEOUT:-1800}
KV_CONTEXT_INPUT_RESERVE=${KV_CONTEXT_INPUT_RESERVE:-4096}
KV_CONTEXT_INPUT_SCALE=${KV_CONTEXT_INPUT_SCALE:-1.05}
KV_CONTEXT_SPEC=${KV_CONTEXT_SPEC:-mtp}
KV_CONTEXT_DRAFT=${KV_CONTEXT_DRAFT:-3}
KV_CONTEXT_CHUNK=${KV_CONTEXT_CHUNK:-1024}
KV_CONTEXT_VISION_TOKENS=${KV_CONTEXT_VISION_TOKENS:-8192}
VISION_SOURCE_REVISION=${VISION_SOURCE_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
VISION_QUALITY_CONTEXT=${VISION_QUALITY_CONTEXT:-32768}
VISION_QUALITY_COMPARISON=${VISION_QUALITY_COMPARISON:-precision} # precision or offload (same BF16 weights)
VISION_QUALITY_KV=${VISION_QUALITY_KV:-bf16}
VISION_QUALITY_SPEC=${VISION_QUALITY_SPEC:-mtp}
VISION_QUALITY_DRAFT=${VISION_QUALITY_DRAFT:-3}
VISION_QUALITY_SEED=${VISION_QUALITY_SEED:-42}
VISION_QUALITY_TOKENS=${VISION_QUALITY_TOKENS:-8192}
VISION_QUALITY_OUTPUT=${VISION_QUALITY_OUTPUT:-1024}
VISION_QUALITY_REPEATS=${VISION_QUALITY_REPEATS:-2}
VISION_QUALITY_TIMEOUT=${VISION_QUALITY_TIMEOUT:-600}
VISION_QUALITY_FONT=${VISION_QUALITY_FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf}
CROSS_LLAMA_BUILD=${CROSS_LLAMA_BUILD:-"$ROOT/build/llamacpp-comparison"}
CROSS_GGUF=${CROSS_GGUF:-"$ROOT/models/comparison/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf"}
CROSS_MMPROJ=${CROSS_MMPROJ:-"$ROOT/models/comparison/unsloth/mmproj-F16.gguf"}

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
        exl3)
            [[ "$VISION" == off ]] || fail "EXL3 comparison is text-only; set VISION=off."
            export CUDA_HOME="$CUDA_ROOT" TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS="$EXL3_BUILD_JOBS"
            export TORCH_EXTENSIONS_DIR="$EXL3_CACHE/torch_extensions" TRITON_CACHE_DIR="$EXL3_CACHE/triton"
            export PATH="$EXL3_ENV/bin:$PATH"
            command=("$EXL3_ENV/bin/python" -u "$EXL3_KIT/tools/serve_openai.py"
                --model "$EXL3_WEIGHTS" --host "$HOST" --port "$PORT"
                --cache_size "$CONTEXT" --grid_size "$EXL3_GPU_GB" --cache_quant "$EXL3_KV")
            files=("$EXL3_ENV/bin/python" "$EXL3_KIT/tools/serve_openai.py" "$EXL3_WEIGHTS/config.json")
            case "$EXL3_SPEC" in
                mtp|none) command+=(--draft_model "$EXL3_SPEC") ;;
                dflash2)
                    command+=(--draft_model "$EXL3_DRAFT_WEIGHTS")
                    files+=("$EXL3_DRAFT_WEIGHTS/config.json") ;;
                *) fail "EXL3_SPEC must be mtp, dflash2, or none." ;;
            esac
            ;;
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
        q27)
            [[ "$VISION" == off ]] || fail "q27 comparison is text-only; set VISION=off."
            [[ "$MTP" != off ]] || fail "q27 comparison uses upstream adaptive speculation; MTP=off is unsupported."
            export Q27_KV Q27_MAXD Q27_PMIN Q27_SUFFIX_W Q27_SUFFIX
            command=("$Q27_SERVER" "$Q27_WEIGHTS" "$Q27_TOKENIZER"
                --host "$HOST" --port "$PORT" --ctx "$CONTEXT" --slots 1
                --no-think --temp 0 --top-p 1 --top-k 0 --min-p 0
                --enable-metrics)
            files=("$Q27_SERVER" "$Q27_WEIGHTS" "$Q27_TOKENIZER")
            ;;
        *) fail "Choose ninfer, llamacpp, q27, or exl3." ;;
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

    if [[ "$backend" == exl3 ]]; then
        need uv
        [[ "${DRY_RUN:-0}" != 1 ]] || fail "EXL3 setup does not support DRY_RUN; review its commands in run.sh."
        [[ -d "$EXL3_SOURCE/.git" && -d "$EXL3_KIT/.git" ]] || fail "Missing EXL3 checkouts; see README.md."
        [[ $(git -C "$EXL3_SOURCE" rev-parse HEAD) == "$EXL3_REVISION" ]] || fail "EXL3 engine revision mismatch."
        [[ $(git -C "$EXL3_KIT" rev-parse HEAD) == "$EXL3_KIT_REVISION" ]] || fail "EXL3 kit revision mismatch."
        if git -C "$EXL3_KIT" apply --check "$ROOT/patches/mia-exl3-benchmark-metrics.patch" 2>/dev/null; then
            git -C "$EXL3_KIT" apply "$ROOT/patches/mia-exl3-benchmark-metrics.patch"
        else
            git -C "$EXL3_KIT" apply --reverse --check "$ROOT/patches/mia-exl3-benchmark-metrics.patch" || fail "EXL3 metrics patch does not match checkout."
        fi
        [[ -x "$EXL3_ENV/bin/python" ]] || uv venv --python 3.11 "$EXL3_ENV"
        uv pip install --python "$EXL3_ENV/bin/python" "torch==$EXL3_TORCH" --index-url "$EXL3_TORCH_INDEX"
        mkdir -p "$EXL3_CACHE"
        printf 'torch==%s\n' "$EXL3_TORCH" > "$EXL3_CACHE/torch-constraint.txt"
        uv pip install --python "$EXL3_ENV/bin/python" -c "$EXL3_CACHE/torch-constraint.txt" \
            -r "$ROOT/requirements-exl3.txt"
        mkdir -p "$EXL3_BUILD"
        if [[ ! -e "$EXL3_SOURCE/build" && ! -L "$EXL3_SOURCE/build" ]]; then
            ln -s "$(realpath --relative-to="$EXL3_SOURCE" "$EXL3_BUILD")" "$EXL3_SOURCE/build"
        fi
        export CUDA_HOME="$CUDA_ROOT" TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS="$EXL3_BUILD_JOBS"
        export PATH="$EXL3_ENV/bin:$PATH"
        uv pip install --python "$EXL3_ENV/bin/python" --no-build-isolation --no-deps "$EXL3_SOURCE"
        uv pip freeze --python "$EXL3_ENV/bin/python" > "$EXL3_CACHE/requirements-resolved.txt"
        return
    fi

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
        q27)
            need make
            [[ -f "$Q27_SOURCE/Makefile" ]] || fail "Missing q27 checkout: $Q27_SOURCE"
            if [[ "${DRY_RUN:-0}" == 1 ]]; then
                quote make -C "$Q27_SOURCE" -j build/q27-server-w8 "NVCC=$CUDA_ROOT/bin/nvcc" "$@"
                return
            fi
            exec make -C "$Q27_SOURCE" -j build/q27-server-w8 "NVCC=$CUDA_ROOT/bin/nvcc" "$@"
            ;;
        *) fail "Choose ninfer, llamacpp, or q27." ;;
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
  ./run.sh quality [--fixtures retrieval-vision ledger-reasoning]
  ./run.sh vision-source # download the pinned official shard containing Vision
  ./run.sh vision-convert # preserve all nonvision payloads; create a BF16-vision artifact
  ./run.sh vision-quality # matched resident quantized/BF16 image comparison
  ./run.sh kv-context # FP8/INT8 incremental capacity and long-context quality, BF16 vision offload
  VISION_QUALITY_COMPARISON=offload ./run.sh vision-quality # same BF16 artifact, resident vs host staging
  ./run.sh setup {ninfer|llamacpp}
  ./run.sh serve {ninfer|llamacpp}
  ./run.sh benchmark
  ./run.sh compare [--modes ninfer-mtp ninfer-dflash5 llamacpp q27] [--vision]
  ./run.sh setup exl3 # existing pinned checkouts; isolated environment and reporting patch
  ./run.sh exl3-download # pinned target and DFlash2 weights
  CONTEXT=32768 VISION=off ./run.sh serve exl3
  ./run.sh compare-exl3 # four profiles; thinking enabled, generous quality output budget

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
    setup) setup "$@" ;;
    exl3-download)
        [[ -x "$EXL3_ENV/bin/hf" ]] || fail "Run ./run.sh setup exl3 first."
        export HF_HOME="$EXL3_CACHE/huggingface"
        "$EXL3_ENV/bin/hf" download Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw --revision "$EXL3_WEIGHTS_REVISION" --local-dir "$EXL3_WEIGHTS"
        exec "$EXL3_ENV/bin/hf" download Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw --revision "$EXL3_DRAFT_REVISION" --local-dir "$EXL3_DRAFT_WEIGHTS"
        ;;
    serve) serve "$@" ;;
    benchmark)
        need uv
        export CONTEXT # fixed long-context workload uses the same capacity for both engines
        export BENCHMARK_WARMUP BENCHMARK_REPEATS BENCHMARK_TOKENS BENCHMARK_TIMEOUT
        exec "$ROOT/benchmark.py"
        ;;
    quality)
        need uv
        export QUALITY_PROFILES QUALITY_OUTPUT QUALITY_SEED QUALITY_TIMEOUT QUALITY_SPEC QUALITY_THINKING_BUDGET
        export QUALITY_INPUT_TOKENS
        export NINFER_WEIGHTS NINFER_VISION_OFFLOAD
        exec uv run --no-project --python 3.11 python "$ROOT/quality_benchmark.py" "$@"
        ;;
    vision-source)
        need hf
        exec hf download Qwen/Qwen3.8-27B model-00001-of-00018.safetensors \
            model.safetensors.index.json config.json preprocessor_config.json \
            --revision "$VISION_SOURCE_REVISION" --local-dir "$VISION_BF16_SOURCE"
        ;;
    vision-convert)
        [[ -x "$VISION_PYTHON" ]] || fail "Missing vision environment; see requirements-vision.txt."
        [[ "$VISION_PYTHON" == /* ]] || VISION_PYTHON="$ROOT/$VISION_PYTHON"
        [[ "$NINFER_WEIGHTS" == /* ]] || NINFER_WEIGHTS="$ROOT/$NINFER_WEIGHTS"
        [[ "$VISION_BF16_SOURCE" == /* ]] || VISION_BF16_SOURCE="$ROOT/$VISION_BF16_SOURCE"
        [[ "$VISION_BF16_WEIGHTS" == /* ]] || VISION_BF16_WEIGHTS="$ROOT/$VISION_BF16_WEIGHTS"
        cd "$NINFER_SOURCE"
        exec "$VISION_PYTHON" -m tools.convert.qwen3_8_27b.vision_bf16 \
            --base "$NINFER_WEIGHTS" --source "$VISION_BF16_SOURCE" --out "$VISION_BF16_WEIGHTS"
        ;;
    vision-quality)
        [[ -x "$VISION_PYTHON" ]] || fail "Missing vision environment; see requirements-vision.txt."
        export NINFER_WEIGHTS VISION_BF16_WEIGHTS VISION_QUALITY_CONTEXT VISION_QUALITY_KV
        export VISION_QUALITY_SPEC VISION_QUALITY_TOKENS VISION_QUALITY_OUTPUT VISION_QUALITY_REPEATS
        export VISION_QUALITY_DRAFT VISION_QUALITY_SEED
        export VISION_QUALITY_COMPARISON
        export VISION_QUALITY_TIMEOUT VISION_QUALITY_FONT
        exec "$VISION_PYTHON" "$ROOT/vision_quality_benchmark.py"
        ;;
    kv-context)
        [[ -x "$VISION_PYTHON" ]] || fail "Missing vision environment; see requirements-vision.txt."
        export VISION_BF16_WEIGHTS QUALITY_OUTPUT
        export KV_CONTEXT_MODES KV_CONTEXT_START KV_CONTEXT_STEP KV_CONTEXT_MAX KV_CONTEXT_HEADROOM_MIB
        export KV_CONTEXT_OUTPUT KV_CONTEXT_SEED KV_CONTEXT_TIMEOUT KV_CONTEXT_INPUT_RESERVE KV_CONTEXT_INPUT_SCALE
        export KV_CONTEXT_SPEC KV_CONTEXT_DRAFT KV_CONTEXT_CHUNK KV_CONTEXT_VISION_TOKENS
        exec "$VISION_PYTHON" "$ROOT/kv_context_benchmark.py"
        ;;
    compare|compare-exl3)
        need uv
        if [[ "$action" == compare-exl3 ]]; then
            CROSS_CONTEXT="$EXL3_CONTEXT"
            CROSS_THINKING=on
            CROSS_QUALITY_TOKENS="$EXL3_COMPARE_QUALITY_TOKENS"
            set -- --modes ninfer-mtp ninfer-dflash5 exl3-mtp exl3-dflash2 "$@"
        fi
        export CROSS_CONTEXT CROSS_REPEATS CROSS_TOKENS CROSS_WARMUP_TOKENS CROSS_TIMEOUT
        export CROSS_MTP_DRAFT CROSS_THINKING CROSS_QUALITY_TOKENS
        export EXL3_SOURCE EXL3_KIT EXL3_ENV EXL3_CACHE EXL3_WEIGHTS EXL3_DRAFT_WEIGHTS
        export EXL3_KV EXL3_GPU_GB EXL3_WEIGHTS_REVISION EXL3_DRAFT_REVISION
        export CROSS_LLAMA_BUILD CROSS_GGUF CROSS_MMPROJ
        export Q27_SOURCE Q27_SERVER Q27_WEIGHTS Q27_TOKENIZER
        exec uv run --no-project --python 3.11 python "$ROOT/cross_benchmark.py" "$@"
        ;;
    help|-h|--help|"") usage ;;
    *) fail "Unknown action: $action" ;;
esac

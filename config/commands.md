# Terminal commands

The main commands are the same in cook-4090 and cook-studio:

| Command | Action |
| --- | --- |
| `./run.sh` | Run host inference with the default configuration. |
| `./run.sh --verify` | Check local setup and resource planning budgets. |
| `./run.sh --benchmark` | Measure installed model/backend combinations sequentially. |

The default starts the host inference API with `CODING_PROFILE=qwen38-exllamav3`.
It starts or updates Docker services, then runs inference in this terminal.
Models load on demand. Docker services remain available after Ctrl+C.
Existing setup, download, and maintenance commands remain available.
Inference stays in the foreground. Engine stdout and stderr appear in the same
terminal. Press Ctrl+C to stop all workers and release the GPU. No host daemon
is installed or started. A duplicate launch checks the existing service and
reports its health instead of starting a second worker.

Image requests reuse the pipeline until coding needs the GPU. Set `IMAGE_REUSE=off`
to release it after each image. Images use INT8 ConvRot transformer linears,
Comfy Kitchen decode attention, and compiled transformer blocks. The text encoder,
VAE, conditioning, and final output projection remain BF16. Masked prefill uses SDPA.
Feed-forward layers process up to `IMAGE_FF_CHUNK_SIZE=4096` tokens per pass to
reduce temporary memory at high resolutions. Set it to `0` to disable chunking.
Set `IMAGE_QUANTIZATION=bf16 IMAGE_ATTENTION=sdpa IMAGE_COMPILE=off IMAGE_FF_CHUNK_SIZE=0`
to use the original image path. `IMAGE_ATTENTION=flex-compiled` selects compiled FlexAttention.
Compilation adds startup time and can repeat when input shapes change.
Compiler caches stay in the ignored `build/qwen-image-cache/` directory.

## Verification

Verification checks Python dependencies, CUDA, engine executables, local weight
files, image shard indexes, RAM, GPU capacity, disk space, private configuration,
and Docker/Compose/Caddy configuration. It does not generate images or tokens.
Private values are not printed. Use `chmod 600 .env` to restrict configuration.

The default planning budgets are 48 GiB available RAM, 22 GiB GPU capacity, and
10 GiB free disk after weights are installed. Change `VERIFY_RAM_GIB`,
`VERIFY_VRAM_GIB`, and `VERIFY_DISK_GIB` in the environment when needed.
These budgets target the configured RTX 4090 deployment with image CPU offload.
They do not prove that every context length, image size, or backend fits.
Weight presence and shard indexes do not prove checksum integrity.
A GPU with little free memory receives a warning because an existing model may
occupy it. Stop other workloads before starting a separate inference process.

Reports use `PASS`, `WARN`, `FAIL`, `SKIP`, and `INFO`. Failures return exit code 1.
A successful verification means static checks passed; it is not a load test.
If the account already belongs to the Docker socket's `docker` group but the
current shell has stale membership, verification refreshes that group for the
command. It does not change account membership or socket permissions.

## Benchmarks

Stop the host inference service before benchmarking. The command takes the same
runtime lock and refuses to interrupt a running service. Missing optional
profiles are skipped. Failed profiles are reported and remaining profiles run.

The workloads come from `config/models.json`: Qwen3.8 with NInfer or ExLlamaV3,
and Qwen-Image-2.1 with Diffusers. Each worker exits before the next
profile starts. There are no benchmark controls in the image workspace.

Defaults in `run.sh`:

- `BENCH_TOKENS=256`, `BENCH_REPEATS=2`, and `BENCH_SEED=42`.
- `BENCH_PROMPT` supplies the same coding workload to each LLM.
- `BENCH_CACHE=cold` starts a new engine for each trial. `warm` runs the exact
  workload once before measurement. The OS file cache is uncontrolled.
- `BENCH_IMAGE_SIZE=1024x1024`, `BENCH_IMAGE_STEPS=4`, and `BENCH_IMAGE_PROMPT`
  supply the same image workload to the image engine. Image trials always
  start a new process. Four steps test speed, not production image quality.

The terminal shows time to first token (TTFT), output tokens per second, actual
output token count, or image generation time. Reports preserve load time,
workload, cache policy, engine settings, token usage, and hardware information.
Missing engine token counts remain unavailable; they are not estimated.
Model loading is excluded from generation timing. Different quantizations and
model implementations make these deployment comparisons, not quality rankings.
The EXL3 profile uses self-calibrated 5-bit weights and Q4 KV cache. NInfer
uses its existing groupwise weights and E8 cache. These formats are different.
See [EXL3 installation](local-inference.md#exllamav3-comparison).

Reports and logs remain in `results/benchmark-*/`, including partial results on
interruption. Restart default inference with `./run.sh` when finished.

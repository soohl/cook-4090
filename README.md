# cook-4090

Qwen3.8-27B experiments on one RTX 4090 (24 GB, SM89), Linux and CUDA 12.8.
All launcher/build/benchmark defaults live in [run.sh](run.sh). Engine implementation
and numerical qualification live in [Ada notes](backends/ninfer/docs/ada.md).
Engine changes are recorded in the submodule revisions; upstream alone does not include them.

## Run

```sh
./run.sh setup ninfer -DBUILD_TESTING=ON
./run.sh serve ninfer
```

Requires the existing CUDA/C++ toolchain and FFmpeg development headers. The default
artifact is `models/dflash2/qwen3_8_27b.ninfer`: bundled Text, MTP, DFlash2 and vision.
Its source is `neroued/Qwen3.8-27B-NInfer`, revision
`dc370fb6295ae8b786e1af4f90d7142a16255c35`. Weights and builds are local and ignored.

Default: 262,144-token capacity, E8 KV, MTP3, vision enabled, 1,024-token prefill
chunks, 8,192 vision tokens and prefix reuse. This is a capacity/performance profile,
not a guarantee of reliable reasoning over every 256K prompt.

For the tested higher-precision KV + BF16 vision profile:

```sh
CONTEXT=139264 NINFER_KV=fp8 NINFER_VISION_OFFLOAD=on \
  NINFER_PREFIX_REUSE=off \
  NINFER_WEIGHTS=models/vision-bf16/qwen3_8_27b.ninfer ./run.sh serve ninfer
```

Use `NINFER_KV=int8` for the matched alternative. Vision offload stages weights
from pinned CPU RAM; vision arithmetic and MTP still run on the GPU. It requires
the BF16-vision artifact and does not increase language-model weight precision.

Other opt-in profiles (qualified at 32K, not 256K):

```sh
CONTEXT=32768 NINFER_SPEC=dflash2 NINFER_DRAFT=5 ./run.sh serve ninfer
CONTEXT=32768 NINFER_DRAFT=5 NINFER_MTP_ADAPTIVE=on ./run.sh serve ninfer
```

DFlash2 replaces MTP; K5 favored the tested code task, K7 structured JSONL.
Adaptive MTP selects greedy depth 3–5 at concurrency one. Fixed MTP3 stays default.

## API and WebUI

NInfer serves its native OpenAI-compatible API at `http://127.0.0.1:8080/v1`.
For llama.cpp's embedded WebUI and API in the same server, use the retained comparison setup:

```sh
./run.sh serve llamacpp
```

This defaults to the local comparison build, Q4_K_XL weights, F16 projector,
fused MTP and 32K context. NInfer retains its 256K default.
WebUI is `/`, API is `/v1`; no service manager or separate UI process is required.
`MTP=inline` uses a fused GGUF head; `on` uses the external draft, `off` disables it.
The launcher builds the embedded UI with gzip disabled for browser compatibility.
Set `HOST=0.0.0.0 PORT=8080` to expose a server on your trusted LAN, then connect to
`http://<host-LAN-IP>:8080`. This exposes an unauthenticated service; do not expose it publicly.

## Experiment lessons

Measurements used one RTX 4090 at its unchanged 350 W limit. These are limited
local workloads, not proof of a hardware ceiling or a globally best engine.

| Area | Retained conclusion |
| --- | --- |
| Ada prefill | Group-64 INT8 dense prefill improved 8K throughput from 2,111 to 3,548 tok/s; later E8/Q4 tuning reached about 3,684. INT8 activation compute is lossy, unlike storage-preserving scheduling changes. |
| Speculation | MTP3 is the balanced choice. DFlash2 K5/K7 and adaptive MTP help selected tasks; larger fixed windows hurt prose. Keep MTP on GPU: disabling it saved about 978 MiB at 32K but reduced measured decode from 163 to 51 tok/s. |
| Engine comparison | NInfer was competitive with q27 and faster than the tested llama.cpp setup. Different weights, KV formats and prompt rendering prevent a pure engine ranking. |
| Prefix reuse | Bounded retention and host-backed state checkpoints cut repeated-prefix TTFT substantially; cache hits are not fresh-prefill throughput. |
| Vision precision | Nine synthetic images: quantized vision scored 150/157 exact fields (4/9 complete answers), BF16 153/157 (6/9). This is encouraging, not broad vision qualification. |
| Vision offload | Saves 755.7 MiB versus resident BF16 vision, or 159.0 MiB versus original quantized vision. All 18 matched responses were text-identical; median encode rose from 68.8 to 94.7 ms, decode unchanged. |
| Long-context quality | Separate memory fit, actual prompt length, retrieval, reasoning, output schema and token budget. Earlier forced thinking cutoffs confounded quality results; no general BF16/FP8/INT8/E8 ranking is established. |
| Rejected tuning | Wider Q5 splitting, larger E8 tiles, direct FP8 PV candidates and larger prefill chunks did not offer a qualified overall improvement. |

### FP8 / INT8 context limits

BF16 vision offload, MTP3, 1,024 prefill chunk, 8,192 vision tokens, prefix reuse off:

| Configured capacity | FP8 free MiB at startup | INT8 free MiB at startup |
| --- | ---: | ---: |
| 128K (131,072) | 789.3 | 687.3 |
| 136K (139,264) | 515.3 | 405.3 |
| 144K (147,456) | 241.3 | 125.3 |

136K was the highest tested 8K step retaining the chosen 256 MiB margin; 144K was
startup-only, not quality-qualified. It is not a proven absolute memory limit.
Eight paired requests at 128K/136K passed exact retrieval (12 codes + 3 image facts)
and ledger reasoning. Actual prompts were 110,903–118,952 tokens, not full capacity.
Greedy seed 42, fresh prefixes and a 16,384 total output allowance were used, without
a forced thinking cutoff. FP8 had more headroom; INT8 used fewer reasoning tokens
on these ledger cases. Neither is a universal quality winner.
Separately, E8 at 256K capacity passed retrieval with a 208,360-token vision prompt;
that does not establish deep-reasoning reliability at 256K.

## Reproduce experiments

```sh
./run.sh compare
./run.sh quality
./run.sh vision-quality
VISION_QUALITY_COMPARISON=offload ./run.sh vision-quality
./run.sh kv-context
```

Review the visible settings first: the older `quality` runner defaults to a forced
thinking budget; the incremental `kv-context` runner does not. Results are generated
under ignored `results/`; `benchmark` now creates a timestamped subdirectory instead
of replacing other experiments. Record actual tokens, cache state, settings and GPU
when comparing runs. Reusable fixtures, tests and conversion tools remain in source.
The legacy `benchmark` action keeps a shared 256K capacity for its fixed long prompt;
it is not a smoke test of the 32K llama.cpp serving preset and may exceed that model's VRAM budget.

Raw historical outputs, the transfer-only probe, conversion-source download and
standalone test/benchmark executables were cleaned up. Runnable weights (including
comparison models), servers, build caches, original checkouts, the vision environment
and BF16 conversion manifest remain. To regenerate source and converted vision weights,
use `./run.sh vision-source` then `./run.sh vision-convert`; the pinned source revision
is visible in `run.sh`. The existing `.venv-vision` uses [requirements-vision.txt](requirements-vision.txt).
Removed test/benchmark executables can be rebuilt from the retained CMake build tree.

Upstream and borrowed-kernel attribution is retained in [Ada notes](backends/ninfer/docs/ada.md#provenance)
and original licenses; the q27 comparison build patch remains in `patches/`.

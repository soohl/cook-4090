# cook-4090

Qwen3.8-27B on one RTX 4090 (24 GB), Linux and CUDA 12.8.
Defaults and pinned revisions live in [run.sh](run.sh).

## Run

```sh
./run.sh setup ninfer
./run.sh serve ninfer
```

Requires the existing CUDA/C++ toolchain, FFmpeg development headers and local
weights at `models/dflash2/qwen3_8_27b.ninfer`.

NInfer defaults to 256K capacity, E8 KV, MTP3, vision and prefix reuse.
API: `http://127.0.0.1:8080/v1`. Capacity does not guarantee reasoning quality at 256K.

Higher-precision KV with the retained BF16-vision artifact:

```sh
CONTEXT=139264 NINFER_KV=fp8 NINFER_VISION_OFFLOAD=on \
  NINFER_PREFIX_REUSE=off \
  NINFER_WEIGHTS=models/vision-bf16/qwen3_8_27b.ninfer ./run.sh serve ninfer
```

## Other backends

```sh
./run.sh serve llamacpp                        # WebUI at /, API at /v1; 32K
VISION=off ./run.sh serve exl3                 # text-only MTP; 32K
VISION=off EXL3_SPEC=dflash2 ./run.sh serve exl3 # text-only DFlash2; 32K
```

EXL3 uses the retained [MiaAI kit](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)
and [engine fork](https://github.com/MiaAI-Lab/exllamav3). Rebuild with
`./run.sh setup exl3`; restore weights with `./run.sh exl3-download`.
Setup expects existing checkouts. Dependencies are pinned in
[requirements-exl3.txt](requirements-exl3.txt).

For LAN access, set `HOST=0.0.0.0`. Servers are unauthenticated; keep them private.

## Layout

- `backends/`: source checkouts.
- `models/`: local weights, including `comparison/mia-exl3/`.
- `build/`: binaries, EXL3 environment and runtime caches.
- `results/`: retained comparison reports.
- `patches/`: reproducible backend patches.

Weights, builds, results and optional checkouts are ignored. Temporary experiment
files were cleaned up; runnable backends remain. No further experiments are planned.

## Findings and references

On the tested 4090 workloads, EXL3 DFlash2 was faster on short text and used less
memory; NInfer was faster on 21K-token prompts. The comparison used 32K capacity,
thinking enabled and two trials. Small correctness checks passed; broad quality
and 256K performance were not compared.

- [Local comparison report](results/cross-engine/20260910T115419.341085Z/comparison-summary.md)
- [Ada implementation, earlier measurements and attribution](backends/ninfer/docs/ada.md)
- `./run.sh help` for retained commands and configuration options.

Preserve upstream licenses: NInfer is Apache-2.0; EXL3 code is MIT and its weights
are Apache-2.0. Original licenses remain with each checkout and model.

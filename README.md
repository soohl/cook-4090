# cook-4090

Qwen3.8-27B on one RTX 4090 with NInfer and llama.cpp.

NInfer is pinned to the RTX 4090 port in `soohl/ninfer`. llama.cpp is pinned
to the tested upstream revision. Clone both submodules with:

```sh
git clone --recurse-submodules https://github.com/soohl/cook-4090.git
# Existing checkout:
git submodule update --init
```

## Weights

Place the model files at these ignored paths:

```text
models/qwen3_8_27b.ninfer
models/Qwen3.8-27B-UD-IQ4_XS.gguf
models/mtp-Qwen3.8-27B-Q4_0.gguf
models/mmproj-Qwen3.8-27B-Q8_0.gguf
```

All build and server flags are visible at the top of `run.sh`. Override a
setting in the environment for one experiment.

## Run

```sh
./run.sh setup ninfer
./run.sh setup llamacpp

./run.sh serve ninfer
./run.sh serve llamacpp
```

Stop one server before starting the other. Both expose `qwen3.8-27b` at
`http://127.0.0.1:8080/v1`.

Defaults: 256K context, one request, vision, and MTP. NInfer uses E8 KV/MTP3.
llama.cpp uses Q4_0 KV/MTP4.

## Benchmark

```sh
./run.sh benchmark
```

The same near-256K request runs through both engines with prompt caching
disabled. A quality pass requires the exact answer
`ORCHID=493817; COLOR=COBALT`. The report also includes TTFT, TPOT, latency,
throughput, tokens, MTP acceptance, versions, and GPU.

Each run replaces:

```text
results/
├── report.json
└── logs/
    ├── ninfer.log
    └── llamacpp.log
```

Set `BENCHMARK_WARMUP=1` and `BENCHMARK_REPEATS=3` in `run.sh` for final
measurements. The retrieval marker checks 256K operation, not broad knowledge.

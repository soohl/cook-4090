# cook-4090 rules

- Target Qwen3.8-27B and Qwen-Image-2.1 on one RTX 4090.
- Use ExLlamaV3 with EXL3 weights by default for coding. Keep NInfer for
  comparison. Use Diffusers for images.
- Keep all build, serve, and benchmark defaults visible in `run.sh`.
- Use project uv environments with uv-managed Python, never system Python.
  Prefer matching prebuilt wheels. Keep separate engine environments when required.
- Keep `./run.sh`, `--verify`, and `--benchmark` consistent with cook-studio.
  Keep benchmarks in the terminal, outside the UI.
- Keep inference and the image app offline at runtime. Only the Tailscale edge
  may use the internet. Keep image sessions until the user deletes them.
  Keep model and engine adapters separate from UI code.
- Run inference on the host. Run the UI, API edge, passkeys, and tunnel in Docker.
  Keep `./run.sh` in the foreground with engine logs. Do not start a host daemon.
  Keep coding port 8000 private. See [stack setup](config/stack.md).
- Use repository-relative paths. Keep weights, builds, captures, and results ignored.
- Keep private deployment values in the ignored `.env`. Read them at runtime.
  Do not copy them into code, scripts, or documentation. Keep the stack lean.
- Put Python code in `src/`, profiles in `config/`, and tests in `tests/`.
- Follow [NInfer instructions](backends/ninfer/AGENTS.md) for engine changes.
- Match benchmark workloads. Record token counts, cache state, settings, and GPU.
- Use STE-inspired technical English in Markdown. Preserve technical meaning.
- Update affected docs with code changes. Keep project guidance short and linked.
- Keep documentation short. Do not add task journals or duplicate command help.
- Check documentation links against a fresh checkout. Label ignored local artifacts.
- Preserve licenses and attribution.
- Do not commit, push, deploy, alter system packages, or remove original
  checkouts without explicit authorization.

# cook-4090 rules

- Target Qwen3.8-27B and Qwen-Image-2.1 on one RTX 4090.
- Support NInfer and llama.cpp for LLMs. Use Diffusers for image generation.
- Keep all build, serve, and benchmark defaults visible in `run.sh`.
- Keep cook-4090 offline at runtime. Clear UI artifacts
  on restart. Keep model and engine adapters separate from UI code.
- Use repository-relative paths. Keep weights, builds, captures, and results ignored.
- Put Python code in `src/`, profiles in `config/`, and tests in `tests/`.
- Follow [NInfer instructions](backends/ninfer/AGENTS.md) for engine changes.
- Match benchmark workloads. Record token counts, cache state, settings, and GPU.
- Use STE-inspired technical English in Markdown. Preserve technical meaning.
- Update affected docs with code changes. Keep project guidance short and linked.
- Keep README.md short. Store detailed local notes in ignored `docs/`.
- Check documentation links against a fresh checkout. Label ignored local artifacts.
- Preserve licenses and attribution.
- Do not commit, push, deploy, alter system packages, or remove original
  checkouts without explicit authorization.

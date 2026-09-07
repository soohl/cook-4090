# cook-4090 rules

- Scope: Qwen3.8-27B on one RTX 4090.
- Keep all build, serve, and benchmark defaults visible in `run.sh`.
- Use repository-relative paths; keep weights, builds, and results ignored.
- Engine changes follow `backends/ninfer/AGENTS.md`.
- Keep weights, builds, captures, and results ignored.
- Match benchmark workloads and record token counts, cache state, settings, and GPU.
- Preserve licenses and attribution.
- Do not commit, push, deploy, alter system packages, or remove original
  checkouts without explicit authorization.

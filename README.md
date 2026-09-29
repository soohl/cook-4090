<p align="center">
  <img src="assets/cook-4090.png" alt="An orange cat chef cooks an RTX 4090 in a flaming frying pan." width="480">
</p>

# cook-4090

A private Qwen3.8-27B API for external coding agents and a Qwen-Image-2.1 image
app on one RTX 4090. Inference runs on the Linux host. Docker runs the API edge,
React image workspace, separate passkey login, and Tailscale tunnel.

- Coding: `http://<tailscale-node>:8000/v1`, with an API key. Port 8000 is private
  to the tailnet and the configured local address. The default is `127.0.0.1`.
- Images: `https://<tailscale-node>`, with cook-4090 passkeys. Optional Funnel
  forwards encrypted TCP to Caddy on this machine. The relay cannot read images
  or prompts. TLS terminates here; local inference must read the request.
- ExLlamaV3 through TabbyAPI handles coding by default. NInfer remains available.
  Diffusers handles images.
  One worker owns the GPU. Requests wait while another request uses it.
- Image sessions keep prompts, settings, references, and results across restarts.
  Open or delete sessions from the left sidebar. Passkeys and tunnel identity persist.

Inference and the image app have no internet access. Only Tailscale needs
external connectivity during runtime. Install dependencies and download models
before use. Defaults and container pins are in [run.sh](run.sh).

## Start

Use Linux, an RTX 4090, a compatible NVIDIA driver, Git, `uv`, Docker Compose,
and `libseccomp.so.2`. Build the selected LLM engine and supply its local weights.
See [model setup and local tools](config/local-inference.md).

```sh
git submodule update --init --recursive
./run.sh image-setup
./run.sh image-download
./run.sh setup exllamav3
./run.sh exl3-download
./run.sh stack-init
```

Follow [stack setup](config/stack.md) to enroll the Tailscale node, create the
separate passkey account, and configure its login client. Then run:

```sh
./run.sh
```

Docker services start automatically. Inference stays in the foreground with
engine logs. Press Ctrl+C to stop inference; Docker services stay available.

Run `./run.sh stack publish` after private access works to enable public image
access. This command never publishes coding port 8000.

## Checks and references

- `./run.sh --verify`: check setup and estimated resource needs without loading models.
- `./run.sh --benchmark`: compare installed model/backend profiles in the terminal.
  Stop host inference first. Reports remain in ignored `results/`.
- [Terminal commands and measurement limits](config/commands.md).
- `./run.sh test`: CPU checks for inference, streams, privacy, and route isolation.
- `./run.sh check`: CPU tests, repository privacy, Compose, and Caddy validation.
- `./run.sh stack check`: Compose and Caddy validation with Docker available.
- `./run.sh help`: commands and settings.
- [Stack operation and access boundaries](config/stack.md).
- [Model installation, benchmarks, and legacy Gradio tools](config/local-inference.md).
- [Model profiles](config/models.json).

Weights, builds, results, credentials, and local `docs/` notes are ignored.
The repository uses [Apache 2.0](LICENSE). Engines and models retain their own
licenses. Qwen-Image-2.1 uses the Qwen Research License.

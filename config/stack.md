# Docker stack

## Boundaries

The host inference service owns the GPU and its child processes. It accepts
requests on `build/stack-ipc/inference.sock`. The socket permits only its owner
and root. The image container uses the host user's numeric UID and GID.
Docker must run on the inference host. Rootless Docker and user namespace
remapping need matching socket permissions and are not tested.
Caddy uses `DAC_OVERRIDE` to connect to the owner-only host socket. A one-time
initializer sets the Pocket ID volume owner before its unprivileged server starts.

Caddy forwards private coding requests to the socket. The image app forwards
only image requests and health checks. Containers do not mount model weights,
engine binaries, the repository, or the Docker socket.

The default coding address is `127.0.0.1:8000`. Set `CODING_BIND` in `.env` to
one assigned RFC1918 IPv4 address for LAN access. Wildcard and public addresses
are rejected. The selected LAN interface replaces the loopback publication.
Do not forward this port on your router. Tailnet clients use the dedicated
Tailscale node's name or IP. Restrict port 8000 to your devices in tailnet grants.

The image and identity containers use an internal Docker network. Tailscale
uses a separate network for external connections. Funnel forwards raw TCP:

- Public port 443 reaches Caddy TLS port 8443 for the image app.
- Public port 8443 reaches Caddy TLS port 9443 for Pocket ID.
- Private tailnet port 8000 reaches the coding API. It never uses Funnel.

Browser TLS 1.3 terminates in Caddy on this machine. Funnel relays cannot read
request content. Traffic between local containers uses HTTP. Inference receives
plaintext through a local Unix socket. This is browser-to-host transport
protection, not encryption from the inference host or encryption at rest.
Coding traffic uses Tailscale encryption on the tailnet. LAN HTTP is plaintext.

## Initial setup

Install Docker with Compose before these steps. Install the host inference
requirements with `./run.sh image-setup`. Build an LLM engine and install weights
as described in [local inference](local-inference.md).

1. Run `./run.sh stack-init`.
2. Run `./run.sh stack enroll`.
3. Open the Tailscale login URL and enroll a dedicated `cook-4090` node.
4. Set `FUNNEL_HOSTNAME` in `.env` to its full name, such as
   `cook-4090.example.ts.net`. Use the exact assigned name, without a trailing dot.
5. Enable HTTPS certificates for your tailnet if they are not enabled.
6. Run `./run.sh stack identity`.
7. From a device connected to your tailnet, open
   `https://<FUNNEL_HOSTNAME>:8443/setup`.
8. Create the cook-4090 administrator account and enroll its passkey.
9. Create a confidential OIDC client in this Pocket ID instance. Use
   `https://<FUNNEL_HOSTNAME>/oauth2/callback` as its callback URL.
10. Copy its client ID and client secret into `OIDC_CLIENT_ID` and
    `OIDC_CLIENT_SECRET` in `.env`.
11. Run `./run.sh` to start Docker services and foreground inference.
12. Test coding and image access from a tailnet device.
13. Run `./run.sh stack check`.
14. Enable Funnel for this node in your tailnet policy when public images are needed.
15. Run `./run.sh stack publish` to publish the image and login TLS listeners.

Complete initial account enrollment over private Tailscale access. Do not
publish an unconfigured Pocket ID instance. The public login hostname and
private enrollment hostname are identical, so the passkey remains valid.
The separate instance has no shared credentials, accounts, cookies, or data
volumes with cook-studio. Account registration is disabled after initial setup.
Only administrators can provision additional accounts. All accounts in this
instance can use images. It is intended for a personal deployment.

Keep `.env` private with mode `600`. Use literal, unquoted `KEY=value` entries.
Do not use shell expressions or variable interpolation. The launcher never
sources this file as shell code. `stack-init` refuses to overwrite an existing
file. Keep persistent secrets stable across restarts.

Pocket ID is the trusted identity authority. OAuth2 Proxy validates its issuer,
signatures, state, nonce, and PKCE challenge. It accepts email claims from this
administrator-controlled instance without requiring email verification.
No external identity provider or email-based account merging is configured.

## Coding clients

Set the client's API base URL to `http://<FUNNEL_HOSTNAME>:8000/v1`. Use
`qwen3.8-27b` as the model ID and `CODING_API_KEY` from `.env` as the API key.
Clients may send `Authorization: Bearer <key>` or `x-api-key: <key>`.
The gateway removes the key before sending the request to inference.
Use the protocol supported by your selected engine:

| Route | Purpose |
| --- | --- |
| `GET /v1/models` | Advertised coding model |
| `POST /v1/chat/completions` | Chat Completions, including streamed tool calls |
| `POST /v1/responses` | Stateless Responses, when supported by the engine |
| `POST /v1/responses/input_tokens` | Responses token count, when supported |
| `POST /v1/messages` | Messages protocol, when supported |
| `POST /v1/messages/count_tokens` | Messages token count, when supported |

TabbyAPI supports OpenAI chat/completions. Select NInfer for the Responses or
Messages protocols. Engine errors pass through unchanged. Supply complete
conversation input on each request. Use automatic tool selection with NInfer;
the installed engine rejects forcing a specific named function.
The service rejects `previous_response_id`,
background execution, and `store: true`. For Responses, it sets `store: false`
when omitted. Loading an image model unloads the coding engine and its cache.
The next coding request reloads it. Tools execute on your client device; this
stack does not execute coding-agent commands or expose a workspace.

The default profile is `qwen38-exllamav3` (TabbyAPI). `UI_CONTEXT` sets context capacity;
its default is 32,768 tokens. Profiles are in [models.json](models.json).
Stop independently launched GPU servers before starting this service.
The legacy Gradio UI and this service share a runtime owner lock by default.
Do not override their runtime directories to run both on the same GPU.

## Images and restart behavior

Open `https://<FUNNEL_HOSTNAME>`. Sign in with the cook-4090 passkey. The image-only
React UI and its FastAPI server run in Docker. Select an
installed image engine, enter a prompt, and optionally attach reference images.
Use image numbers in upload order when describing edits. Images and references
pass only between your browser and this machine.

Select **Edit area** below a reference to paint a region. That reference becomes
the base image. The app marks the region on a copy of that reference for Qwen,
then composites the result over the resized original only inside the selection.
This avoids the GPU memory cost of another reference. Region guidance does not
guarantee identity matching.
Use **Edge softness** to blend inside the selection; zero gives a hard boundary.
Selections stay with saved generations. Mask processing and storage run in Docker.

Consecutive image requests reuse the loaded pipeline. Coding requests and host
shutdown stop the image worker and release its memory. Set `IMAGE_REUSE=off`
to release it after every image. The host removes
successful image artifacts after returning the result. The image app saves sessions in the persistent Docker volume `image-history`.
This volume contains prompts, settings, images, and reference uploads.
Keep this volume when rebuilding containers. Delete sessions from the left sidebar.
Engine logs appear in the foreground inference terminal. Benchmark logs are
saved with benchmark results. Failed image artifacts remain in the disposable
runtime until host restart. Session thumbnails restore each image and its prompt
and settings. Saved sessions survive both host and image-app restarts.
Downloaded images are outside this cleanup policy.

Accepted image requests continue when you reload the page or switch sessions.
The Docker build compiles React with Vite. The runtime serves static files and
uses no Node server or external assets.

The sidebar shows numbered reference previews and live inference status, including
the active engine, request time, and queue.

Requests share one GPU queue. A running coding stream finishes before an image
request loads its model. The queue returns HTTP 429 when full. Images do not
cancel active coding work. Long requests can wait for model loading and earlier
work. Adjust queue and timeout defaults in [run.sh](../run.sh) if needed.
The image HTTP connection limit is separate from the GPU queue. Its default is
256 because static assets, API requests, and idle auth proxy connections share
the limit. A low limit can return HTTP 503 and prevent the page from loading.

## Operation

- `./run.sh stack prepare` pulls and builds containers before enrollment.
- `./run.sh` starts or updates Docker services, then runs foreground inference.
- `./run.sh stack status` shows containers, Tailscale, and forwarding rules.
- `./run.sh stack unpublish` changes both public TLS ports back to private Serve.
- `./run.sh stack down` stops containers and retains credentials and identities.
- Stop the host inference terminal separately to release the GPU and clear its
  disposable runtime. A stale runtime is also cleaned at the next host start.
- Run `./run.sh` manually when needed. It does not create a background host
  service. Engine logs appear in that terminal. Press Ctrl+C to stop all workers.
- `./run.sh legacy-ui` runs the optional Gradio tools on loopback port 7860.

Tailscale forwarding settings persist in its state volume. A container restart
can restore a previously enabled Funnel. Use `stack unpublish` before stopping
containers when the next start must remain private. Do not delete identity
volumes as a routine reset. Back up `.env`, `pocket-id-data`, `tailscale-state`,
`image-history`, and Caddy certificate state privately if you need recovery.

`stack up` starts with private image access on first use. Later calls preserve
the existing publication state. Use `stack unpublish` to return to private access.
Docker logs rotate with the `local` driver. Defaults retain three files of up
to 10 MB per container. Set `STACK_LOG_MAX_SIZE` and `STACK_LOG_MAX_FILES` in
the environment to change these limits.

## Validation

Run `./run.sh test` for CPU checks. Run `./run.sh stack check` with the Tailscale
container running to validate Compose and Caddy. Then verify all of these with
real clients before relying on remote access:

1. Confirm that an unauthorized coding request returns HTTP 401.
2. Confirm that a tailnet coding request streams tool calls correctly.
3. Confirm that a non-tailnet device cannot connect to port 8000.
4. Confirm that public image access requires the separate passkey.
5. Generate an image and confirm that coding works after the model switch.
6. Restart the image app and confirm that saved sessions reopen.

CPU tests use fake inference. The optional [browser check](../tests/browser_images.py)
also mocks inference and requires Playwright in a separate test environment.
Pass `--connections 256` to match the default `IMAGE_WEB_CONNECTIONS` in `run.sh`.
The check holds 32 extra sockets open to test loading with an idle proxy pool.
Set `CADDY_TEST_BINARY` to a local Caddy executable when running `./run.sh test`
to include the real API-edge authentication and route checks.
Run `COOK_STACK_LIVE=1 ./run.sh test` against an enrolled, running stack to check
Docker permissions, network isolation, private API keys, and image login redirects.
These checks use the private `.env` without printing credentials.
They also check TLS 1.3, rejection of TLS 1.2, raw TCP forwarding, bounded
container logs, and the absence of a public coding Funnel route.
Live Docker networking, passkey enrollment, Funnel, and GPU inference need the
integration checks above on a configured host.

Keep private deployment configuration in the ignored root `.env` (mode 600).
Scripts must read this file instead of copying credentials or assigned hostnames.
Keep only empty credential fields and generic defaults in `env.example`.
Docker volumes retain runtime identity state, including passkeys and TLS keys.
Repository tests check ignored private files and reject copied deployment values
in the publishable tree. Run `./run.sh check` for CPU tests and stack validation.
The image login proxy waits for the web server’s `/healthz` endpoint.
This readiness check does not load a model or claim that host inference is ready.

## References

- [Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve).
- [Tailscale Funnel](https://tailscale.com/docs/reference/tailscale-cli/funnel).
- [Pocket ID installation](https://pocket-id.org/docs/setup/installation).
- [OAuth2 Proxy configuration](https://oauth2-proxy.github.io/oauth2-proxy/configuration/overview/).

The Caddy, Tailscale, and Pocket ID pattern and image pins follow the adjacent
cook-studio project. This stack has its own persistent state and authentication.

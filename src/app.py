"""Gradio composition. Engine orchestration stays in small independent modules."""

import atexit
import os
import signal
import time

from . import ROOT, benchmarks, client, images
from .manager import Manager
from .offline import restrict_ui_network
from .registry import availability, choices, load_registry
from .runtime import Runtime


def build_ui(manager):
    import gradio as gr

    chat_models = choices(manager.registry, "chat")
    image_models = choices(manager.registry, "image")

    def image_sizes(key):
        sizes = images.size_choices(manager.registry[key]) if key else []
        return gr.Dropdown(choices=sizes, value=sizes[0] if sizes else None)

    def load(key, context):
        with manager.lock:
            try:
                manager.ensure_chat(key, int(context))
                return manager.status()
            except Exception as exc:
                raise gr.Error(str(exc)) from None

    def unload():
        with manager.lock:
            manager.stop()
        return manager.status()

    def chat(text, history, key, context, tokens, temperature, thinking):
        if not text.strip():
            raise gr.Error("Enter a message")
        if not 1 <= int(tokens) < int(context):
            raise gr.Error("Output limit must be smaller than the context capacity")
        messages = list(history or []) + [{"role": "user", "content": text}]
        shown = messages + [{"role": "assistant", "content": "Loading model…"}]
        yield shown, history or [], "", "", "Loading selected model…"
        with manager.lock:
            try:
                manager.ensure_chat(key, int(context))
                body = client.payload(manager.model, messages, tokens, temperature, thinking, os.environ["UI_BENCH_SEED"])
                emitted = 0
                for event in client.stream(manager.url, body, int(os.environ["UI_REQUEST_TIMEOUT"])):
                    if not event["done"] and time.monotonic() - emitted < 0.08:
                        continue
                    emitted = time.monotonic()
                    answer = event["text"] or ("Thinking…" if not event["done"] else "No final answer within the output limit. Increase the limit or disable thinking.")
                    current = messages + [{"role": "assistant", "content": answer}]
                    status = manager.status()
                    if event["done"]:
                        status += f" · {event['usage'].get('completion_tokens', '?')} output tokens · {event['elapsed']:.1f}s"
                    yield current, current if event["done"] else history or [], "", event["reasoning"], status
            except GeneratorExit:
                manager.stop()
                raise
            except Exception as exc:
                manager.stop()
                raise gr.Error(str(exc)) from None

    def image(key, prompt, size, steps, seed, references):
        try:
            result, status, report = images.generate(manager, key, prompt, size, steps, seed, references)
            return result, status, report, manager.status()
        except Exception as exc:
            raise gr.Error(str(exc)) from None

    def benchmark(keys, prompt, tokens, repeats, cache, thinking, context):
        try:
            for rows, status, files in benchmarks.run(manager, keys, prompt, tokens, repeats, cache, thinking, context):
                yield rows, status, files, manager.status()
        except Exception as exc:
            raise gr.Error(str(exc)) from None

    with gr.Blocks(title="cook-4090", analytics_enabled=False) as demo:
        gr.Markdown('''<div class="brand">
          <img src="/assets/cook-4090.png" alt="cook-4090 cat chef" width="112" height="112">
          <div><h1>cook-4090</h1><p>Chat, create images, and compare local engines.</p>
          <p>Everything clears when the server restarts. Download anything you want to keep.</p></div>
        </div>''')
        with gr.Row():
            active = gr.Textbox(value=manager.status(), label="GPU", interactive=False, scale=4)
            release = gr.Button("Unload model", scale=1)
        with gr.Tabs():
            with gr.Tab("Chat"):
                history = gr.State([])
                with gr.Row():
                    model = gr.Dropdown(chat_models, value=chat_models[0][1] if chat_models else None, label="Model / engine")
                    load_button = gr.Button("Load model")
                with gr.Accordion("Generation settings", open=False):
                    with gr.Row():
                        context = gr.Number(value=int(os.environ["UI_CONTEXT"]), precision=0, minimum=1024, label="Context capacity")
                        tokens = gr.Number(value=int(os.environ["UI_MAX_TOKENS"]), precision=0, minimum=1, label="Maximum output tokens")
                        temperature = gr.Slider(0, 2, value=float(os.environ["UI_TEMPERATURE"]), step=0.05, label="Temperature")
                        thinking = gr.Checkbox(value=os.environ["UI_THINKING"] == "on", label="Thinking")
                conversation = gr.Chatbot(label="Conversation", height=480, allow_tags=False, buttons=["copy", "copy_all"])
                with gr.Accordion("Reasoning for latest reply", open=False):
                    reasoning = gr.Markdown()
                prompt = gr.Textbox(label="Message", placeholder="Ask something…", lines=2)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    cancel_chat = gr.Button("Stop")
                    clear = gr.Button("New chat")
                chat_inputs = [prompt, history, model, context, tokens, temperature, thinking]
                chat_outputs = [conversation, history, prompt, reasoning, active]
                send_event = send.click(chat, chat_inputs, chat_outputs, api_name="chat", concurrency_id="gpu", concurrency_limit=1)
                enter_event = prompt.submit(chat, chat_inputs, chat_outputs, api_name=False, concurrency_id="gpu", concurrency_limit=1)
                cancel_chat.click(fn=None, cancels=[send_event, enter_event], queue=False)
                clear.click(lambda: ([], [], "", ""), outputs=[conversation, history, prompt, reasoning],
                            cancels=[send_event, enter_event], queue=False, api_name="clear_chat")
                model.change(load, [model, context], active, concurrency_id="gpu", api_name=False)
                load_button.click(load, [model, context], active, concurrency_id="gpu", api_name="load_model")
            with gr.Tab("Images"):
                image_model = gr.Dropdown(image_models, value=image_models[0][1] if image_models else None, label="Model / engine")
                with gr.Row():
                    with gr.Column():
                        image_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe an image, or what to change in a reference image.")
                        refs = gr.File(label="Reference images (optional)", file_count="multiple", file_types=["image"], type="filepath", allow_reordering=True)
                        gr.Markdown("Upload an image to edit it. For multiple references, use ‘image 1’, ‘image 2’, etc. in upload order.")
                        sizes = images.size_choices(manager.registry[image_models[0][1]]) if image_models else []
                        size = gr.Dropdown(sizes, value=sizes[0] if sizes else None, label="Image size (width × height)",
                                           info="Resolution presets for the selected model. Larger images take longer and use more GPU memory.")
                        with gr.Accordion("Settings", open=False):
                            steps = gr.Slider(1, int(os.environ["IMAGE_UI_MAX_STEPS"]), value=int(os.environ["IMAGE_STEPS"]), step=1, label="Steps")
                            seed = gr.Number(value=int(os.environ["IMAGE_SEED"]), precision=0, label="Seed (-1 for random)")
                        generate = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        preview = gr.Image(label="Result", type="filepath", interactive=False, format="png")
                        image_status = gr.Textbox(label="Generation", interactive=False)
                        image_report = gr.File(label="Download generation details", interactive=False)
                generate.click(image, [image_model, image_prompt, size, steps, seed, refs],
                               [preview, image_status, image_report, active], api_name="generate_image", concurrency_id="gpu", concurrency_limit=1)
                image_model.change(image_sizes, image_model, size, queue=False, api_name=False)
            with gr.Tab("Benchmarks"):
                gr.Markdown("Compare the same workload across model/engine profiles. Model loading is measured separately. Cold starts use a fresh engine; warm runs first repeat the exact workload. Settings and quantization differences are included in the export.")
                models = gr.Dropdown(chat_models, value=[p[1] for p in chat_models[:2]], multiselect=True, label="Models / engines to compare")
                workload = gr.Textbox(value=os.environ["UI_BENCH_PROMPT"], label="Matched workload", lines=4)
                with gr.Row():
                    bench_tokens = gr.Number(value=int(os.environ["UI_BENCH_TOKENS"]), precision=0, minimum=1, label="Maximum output tokens")
                    repeats = gr.Number(value=int(os.environ["UI_BENCH_REPEATS"]), precision=0, minimum=1, maximum=10, label="Trials per engine")
                    cache = gr.Dropdown([("Cold: fresh engine", "cold"), ("Warm: exact repeat", "warm")], value=os.environ["UI_BENCH_CACHE"], label="Cache state")
                    bench_thinking = gr.Checkbox(value=os.environ["UI_THINKING"] == "on", label="Thinking for all models")
                    bench_context = gr.Number(value=int(os.environ["UI_CONTEXT"]), precision=0, minimum=1024, label="Context capacity")
                with gr.Row():
                    run = gr.Button("Run comparison", variant="primary")
                    cancel_bench = gr.Button("Stop comparison")
                table = gr.Dataframe(headers=benchmarks.HEADERS, interactive=False, label="Measurements")
                bench_status = gr.Textbox(label="Progress", interactive=False)
                exports = gr.File(file_count="multiple", label="Download results (JSON + CSV)", interactive=False, height=110)
                bench_event = run.click(benchmark, [models, workload, bench_tokens, repeats, cache, bench_thinking, bench_context],
                                       [table, bench_status, exports, active], api_name="benchmark", concurrency_id="gpu", concurrency_limit=1)
                cancel_bench.click(fn=None, cancels=[bench_event], queue=False)
            with gr.Tab("Models"):
                gr.Markdown("Installed profiles are configured locally in `config/models.json`. Switching unloads the previous engine. The UI never downloads models or installs software.")
                gr.Dataframe(headers=["Profile", "Type", "Engine", "Availability"],
                             value=[[p["label"], p["kind"], p["engine"], availability(p)] for p in manager.registry.values()], interactive=False)
        release.click(unload, outputs=active, concurrency_id="gpu", api_name="unload_model")
    return demo


def main():
    runtime = Runtime(os.environ["UI_RUNTIME"])
    os.environ.update(GRADIO_TEMP_DIR=str(runtime.path / "gradio"), TMPDIR=str(runtime.path))
    restrict_ui_network()
    manager = Manager(load_registry(os.environ["UI_REGISTRY"]), runtime.path)

    def cleanup():
        manager.close()
        runtime.close()

    atexit.register(cleanup)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    print(f"cook-4090: http://{os.environ['UI_HOST']}:{os.environ['UI_PORT']} · runtime {runtime.path}", flush=True)
    # All stack logs, including dependency tracebacks, are ephemeral too.
    log = (runtime.path / "server.log").open("a", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)

    import gradio as gr
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import FileResponse, PlainTextResponse

    class LocalBrowser(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path == "/session-epoch":
                response = PlainTextResponse(runtime.path.name)
            elif request.url.path == "/assets/cook-4090.png":
                response = FileResponse(ROOT / "assets/cook-4090.png", media_type="image/png")
            else:
                response = await call_next(request)
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; connect-src 'self'; img-src 'self' data: blob:; "
                "media-src 'self' data: blob:; font-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; worker-src 'self' blob:; "
                "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'self'"
            )
            response.headers["Cache-Control"] = "no-store"
            return response

    # A still-open browser clears its rendered history when it observes a restart.
    head = """<script>
try {for (const key of Object.keys(localStorage)) if(key.startsWith('gradio:run-history:')) localStorage.removeItem(key)} catch {}
setInterval(async()=>{try{const r=await fetch('/session-epoch',{cache:'no-store'});if(r.ok&&(await r.text())!==EPOCH)location.reload()}catch{}},5000)
</script>""".replace("EPOCH", repr(runtime.path.name))
    demo = build_ui(manager)
    demo.queue(max_size=int(os.environ["UI_QUEUE"]), default_concurrency_limit=1).launch(
        server_name=os.environ["UI_HOST"], server_port=int(os.environ["UI_PORT"]),
        share=False, inbrowser=False, show_error=False, enable_monitoring=False, ssr_mode=False,
        run_history=False, footer_links=["api"],
        favicon_path=str(ROOT / "assets/cook-4090.png"),
        css=".brand {display:flex; align-items:center; gap:20px; flex-wrap:wrap} "
            ".brand img {border-radius:12px; flex-shrink:0} "
            ".brand div {flex:1; min-width:220px} "
            ".brand h1 {font-size:2rem; font-weight:700; margin:0 0 6px} "
            ".brand p {margin:4px 0}",
        theme=gr.themes.Base(font=["system-ui", "sans-serif"], font_mono=["monospace"]),
        head=head, app_kwargs={"middleware": [Middleware(LocalBrowser)]},
    )


if __name__ == "__main__":
    main()

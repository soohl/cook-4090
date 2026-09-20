"""Small streaming OpenAI-compatible HTTP client; no inference/SDK dependencies."""

import json
import time
import urllib.error
import urllib.request


def stream(url, payload, timeout):
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    text = reasoning = ""
    usage, timings = {}, {}
    first = None
    finish = None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Engine HTTP {exc.code}: {exc.read().decode()[:1000]}") from None
    with response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                elapsed = time.perf_counter() - started
                tokens = usage.get("completion_tokens")
                yield dict(text=text, reasoning=reasoning, usage=usage, timings=timings, done=True,
                           ttft=first, elapsed=elapsed, finish_reason=finish,
                           tokens_per_second=(tokens - 1) / (elapsed - first)
                           if tokens and tokens > 1 and first is not None and elapsed > first else None)
                return
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if (content or thought) and first is None:
                    first = time.perf_counter() - started
                text += content
                reasoning += thought
                finish = choice.get("finish_reason") or finish
            yield dict(text=text, reasoning=reasoning, done=False)
    raise RuntimeError("Engine disconnected before completing the response")


def payload(model, messages, tokens, temperature, thinking, seed):
    result = dict(model=model, messages=messages, max_tokens=int(tokens), temperature=float(temperature),
                  seed=int(seed), stream=True, stream_options={"include_usage": True},
                  chat_template_kwargs={"enable_thinking": bool(thinking)}, cache_prompt=True)
    if not thinking:
        result["reasoning_effort"] = "none"
    return result

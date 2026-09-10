"""Run with the optional EXL3 environment after applying the reporting patch."""
import importlib.util
import json
from pathlib import Path
import unittest

KIT = Path(__file__).resolve().parent / "backends/mia-exl3/tools/serve_openai.py"


@unittest.skipUnless(KIT.exists() and importlib.util.find_spec("aiohttp"),
                     "optional patched EXL3 checkout and environment required")
class StreamingMetricsTest(unittest.IsolatedAsyncioTestCase):
    async def test_stream_reports_same_usage_as_nonstream(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        spec = importlib.util.spec_from_file_location("mia_server_metrics_test", KIT)
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        counters = {"cached_tokens": 128, "time_prefill": 0.1, "time_generate": 0.2}

        def generate(*args, **kwargs):
            if kwargs.get("on_text"):
                kwargs["on_text"]('Checking.</think>\n{"ok":true}')
            return ('Checking.</think>\n{"ok":true}', [], "stop", 256, 12,
                    "Checking.", '{"ok":true}', counters)

        server.generate_full = generate
        app = web.Application()
        app.update(generator=object(), tokenizer=object(), max_body_mb=64)
        app.router.add_post("/v1/chat/completions", server.chat_completions)
        async with TestClient(TestServer(app)) as client:
            payload = {"messages": [{"role": "user", "content": "Check."}],
                       "max_tokens": 32}
            response = await client.post("/v1/chat/completions", json=payload)
            plain = await response.json()
            response = await client.post("/v1/chat/completions", json=dict(
                payload, stream=True, stream_options={"include_usage": True}))
            raw = await response.text()
            events = [json.loads(line[6:]) for line in raw.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            self.assertIn("data: [DONE]", raw)
            usage = next(event for event in events if "usage" in event)
            self.assertEqual(usage["usage"], plain["usage"])
            self.assertEqual(usage["usage"]["completion_tokens"], 12)
            self.assertEqual(usage["usage"]["prompt_tokens_details"]["cached_tokens"], 128)
            self.assertEqual(usage["timings"], counters)
            content = "".join(choice["delta"].get("content", "")
                              for event in events for choice in event.get("choices", []))
            self.assertEqual(content.strip(), plain["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main()

"""Optional live browser regression check; needs Playwright in a separate test venv.

Usage: python tests/browser_app.py --url http://127.0.0.1:7860
Does not load a GPU model. Uses a fresh browser session and disposable canaries.
"""

import argparse
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    origin_host = urlparse(args.url).hostname
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-background-networking"])
        page = browser.new_page()
        errors, external = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: external.append(request.url)
                if urlparse(request.url).hostname not in {origin_host, None} else None)
        page.goto(args.url, wait_until="domcontentloaded")
        expect(page).to_have_title("cook-4090")
        expect(page.get_by_role("heading", name="cook-4090", exact=True)).to_be_visible()
        hero = page.get_by_role("img", name="cook-4090 cat chef", exact=True)
        expect(hero).to_be_visible()
        page.wait_for_function("document.querySelector('.brand img')?.naturalWidth > 0")
        asset = page.request.get(args.url + "/assets/cook-4090.png")
        assert asset.ok and asset.headers["content-type"].startswith("image/png")
        assert asset.headers["cache-control"] == "no-store"
        assert page.request.get(args.url + "/session-epoch").ok
        expect(page.get_by_role("button", name="New chat", exact=True)).to_be_visible()
        config = page.request.get(args.url + "/config").json()
        assert config["run_history"] is False
        # Simulate storage left by an earlier Gradio release/app lifetime.
        page.evaluate("localStorage.setItem('gradio:run-history:v2:legacy-test', 'private-canary')")
        page.reload(wait_until="domcontentloaded")
        expect(page.get_by_role("button", name="New chat", exact=True)).to_be_visible()
        assert not page.evaluate("Object.keys(localStorage).filter(k=>k.startsWith('gradio:run-history:'))")
        page.get_by_label("Message", exact=True).fill("unsent-private-canary")
        page.get_by_role("button", name="New chat", exact=True).click()
        expect(page.get_by_label("Message", exact=True)).to_have_value("")
        assert not page.evaluate("Object.keys(localStorage).filter(k=>k.startsWith('gradio:run-history:'))")
        for tab in ("Images", "Benchmarks", "Models", "Chat"):
            page.get_by_role("tab", name=tab, exact=True).click()
        assert not errors, errors
        assert not external, external
        print("PASS: cook-4090 branding and local artwork, UI sections, history cleanup, no external requests")
        browser.close()


if __name__ == "__main__":
    main()

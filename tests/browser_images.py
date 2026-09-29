"""Browser regression checks for the React workspace with fake host inference."""

import argparse
import base64
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.request

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-python', required=True)
    parser.add_argument('--screenshot')
    parser.add_argument('--connections', type=int, default=256)
    args = parser.parse_args()
    requests, available = [], [True]
    edited_png = [PNG]
    health = dict(status='ok', busy=False, queued=0, engine=None, active_model=None, elapsed_seconds=0)

    class Inference(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, payload, status=200):
            content = json.dumps(dict(payload, epoch='host')).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if not available[0]:
                self.send({'detail': 'Inference is offline'}, 503)
            elif self.path == '/images/options':
                self.send(dict(models=[dict(id='image', label='Qwen-Image-2.1 · Diffusers', sizes=['1024x1024', '2048x2048'])], steps=40, max_steps=50, max_references=10))
            else:
                self.send(health)

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if requests[-1]['prompt'] == 'A sunlit ocean':
                time.sleep(3)
            self.send(dict(image=edited_png[0] if 'green-painted area' in requests[-1]['prompt'] else PNG, status='1024x1024 · 40 steps · seed 42'))

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
        uds = str(Path(directory) / 'inference.sock')
        server = Server(uds, Inference)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        listener.bind(('127.0.0.1', 0)); listener.listen()
        origin = f'http://127.0.0.1:{listener.getsockname()[1]}'
        env = dict(os.environ, INFERENCE_SOCKET=uds, UI_QUEUE='2', STACK_MAX_BODY='33554432', IMAGE_WEB_TIMEOUT='30', IMAGE_HISTORY_DIR=str(Path(directory) / 'history'))
        command = [args.server_python, '-m', 'uvicorn', 'src.image_web:app', '--fd', str(listener.fileno()), '--no-access-log', '--limit-concurrency', str(args.connections)]
        def start():
            return subprocess.Popen(command, cwd=ROOT, env=env, pass_fds=(listener.fileno(),), stdout=subprocess.DEVNULL)
        process = start()
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    urllib.request.urlopen(origin, timeout=.5).close(); break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Image UI did not start')
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=['--disable-background-networking'])
                page = browser.new_page(viewport={'width': 1440, 'height': 1000}, color_scheme='light')
                expect.set_options(timeout=12000)
                errors, external = [], []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('request', lambda request: external.append(request.url) if not request.url.startswith((origin, 'blob:', 'data:')) else None)
                page.goto(origin)
                expect(page.get_by_label('Inference status')).to_contain_text('Inference online')
                health.update(busy=True, engine='diffusers', progress=None)
                progress = page.get_by_role('progressbar', name='Inference progress')
                expect(progress).to_be_visible()
                expect(progress).not_to_have_attribute('value')
                health['progress'] = dict(completed=12, total=25)
                expect(progress).to_have_attribute('value', '12')
                expect(progress).to_have_attribute('max', '25')
                expect(page.get_by_label('Inference status')).not_to_contain_text('elapsed')
                health['progress'] = dict(completed=25, total=25)
                expect(progress).not_to_have_attribute('value')
                health.update(busy=False, engine=None, progress=None)
                expect(progress).to_have_count(0)
                expect(page.get_by_role('combobox', name='Image size', exact=True)).to_be_visible()
                expect(page.locator('details')).to_have_count(0)
                expect(page.get_by_text('Images and prompts stay saved until you delete the session.', exact=True)).to_have_count(0)
                expect(page.get_by_text('Image workspace', exact=True)).to_have_count(0)
                page.get_by_label('Prompt', exact=True).fill('A quiet forest')
                page.get_by_label('Upload reference images').set_input_files({'name':'reference.png', 'mimeType':'image/png', 'buffer':base64.b64decode(PNG)})
                expect(page.get_by_role('button', name='Enlarge reference 1')).to_be_visible()
                page.get_by_role('button', name='Enlarge reference 1').click()
                expect(page.get_by_role('dialog')).to_be_visible()
                page.get_by_role('button', name='Close', exact=True).click()
                page.get_by_role('button', name='Generate', exact=True).click()
                expect(page.locator('.main-image')).to_be_visible()
                expect(page.get_by_label('Saved sessions')).to_contain_text('A quiet forest')
                assert requests[0]['references'] == [PNG]
                saved_url = page.locator('.main-image').get_attribute('src')
                assert page.request.get(origin + saved_url).body().startswith(b'\x89PNG')
                page.get_by_label('Prompt', exact=True).fill('A sunlit ocean')
                page.get_by_role('button', name='Generate', exact=True).click()
                expect(page.get_by_role('status')).to_contain_text('browse other sessions')
                expect(page.locator('.main-image')).to_have_attribute('src', saved_url)
                # A reload during generation must not cancel or lose the result.
                page.reload()
                expect(page.get_by_role('button', name='View image 2', exact=True)).to_be_visible()
                expect(page.locator('.image-caption')).to_contain_text('A sunlit ocean')
                page.get_by_role('button', name='View image 1', exact=True).click()
                expect(page.get_by_label('Prompt', exact=True)).to_have_value('A quiet forest')
                expect(page.get_by_role('button', name='Enlarge reference 1')).to_be_visible()
                stage, rail = page.locator('.image-stage').bounding_box(), page.locator('.image-rail').bounding_box()
                assert stage['x'] + stage['width'] <= rail['x']
                if args.screenshot:
                    page.screenshot(path=args.screenshot, full_page=True)
                page.get_by_role('button', name='New session', exact=True).click()
                expect(page.locator('.main-image')).to_have_count(0)
                page.get_by_label('Prompt', exact=True).fill('Another project')
                page.get_by_role('button', name='Generate', exact=True).click()
                expect(page.locator('.main-image')).to_be_visible()
                expect(page.get_by_role('button', name='Generate', exact=True)).to_be_enabled()
                page.goto('about:blank')
                process.terminate(); process.wait(timeout=10)
                available[0] = False
                process = start()
                page.goto(origin)
                expect(page.get_by_label('Inference status')).to_contain_text('Inference offline')
                expect(page.locator('.main-image')).to_be_visible()
                page.get_by_label('Saved sessions').get_by_role('button', name='A quiet forest').click()
                expect(page.get_by_role('button', name='View image 2', exact=True)).to_be_visible()
                expect(page.get_by_role('button', name='Enlarge reference 1')).to_be_visible()
                available[0] = True
                expect(page.get_by_label('Inference status')).to_contain_text('Inference online')
                page.get_by_role('button', name='Generate', exact=True).click()
                expect(page.get_by_role('button', name='View image 3', exact=True)).to_be_visible()
                assert requests[-1]['references'] == [PNG]
                page.get_by_role('button', name='Delete session', exact=True).click()
                page.get_by_role('button', name='Delete permanently', exact=True).click()
                expect(page.get_by_label('Saved sessions').get_by_role('button', name='A quiet forest')).to_have_count(0)
                assert page.request.get(origin + saved_url).status == 404
                page.get_by_label('Saved sessions').get_by_role('button', name='Another project').click()
                expect(page.locator('.main-image')).to_be_visible()
                # Paint a mask on the second reference, then exercise save/reload/reordering.
                page.get_by_role('button', name='New session', exact=True).click()
                def solid_image(color):
                    return page.evaluate('''color => { const c = document.createElement('canvas'); c.width = c.height = 1024;
                        const ctx = c.getContext('2d'); ctx.fillStyle = color; ctx.fillRect(0, 0, 1024, 1024);
                        return c.toDataURL('image/png').split(',')[1]; }''', color)
                red, blue = solid_image('red'), solid_image('blue')
                edited_png[0] = solid_image('green')
                page.get_by_label('Upload reference images').set_input_files([
                    {'name': 'red.png', 'mimeType': 'image/png', 'buffer': base64.b64decode(red)},
                    {'name': 'blue.png', 'mimeType': 'image/png', 'buffer': base64.b64decode(blue)},
                ])
                page.get_by_role('button', name='Edit area of image 2', exact=True).click()
                expect(page.get_by_role('button', name='Use selection')).to_be_disabled()
                canvas = page.get_by_label('Paint edit area')
                expect(canvas).to_have_attribute('width', '1024')
                bounds = canvas.bounding_box()
                cx, cy = bounds['x'] + bounds['width'] / 2, bounds['y'] + bounds['height'] / 2
                page.mouse.move(cx, cy); page.mouse.down(); page.mouse.move(cx + 30, cy, steps=8); page.mouse.up()
                expect(page.get_by_role('button', name='Use selection')).to_be_enabled()
                page.get_by_role('button', name='Undo selection').click()
                expect(page.get_by_role('button', name='Use selection')).to_be_disabled()
                page.mouse.click(cx, cy)
                expect(page.get_by_label('Edge softness')).to_have_value('12')
                page.get_by_label('Edge softness').focus()
                page.keyboard.press('End')
                expect(page.get_by_label('Edge softness')).to_have_value('40')
                if args.screenshot:
                    page.screenshot(path=args.screenshot + '.mask.png', full_page=True)
                page.get_by_role('button', name='Use selection').click()
                expect(page.get_by_role('button', name='Edit area of image 2')).to_have_text('Area selected')
                page.get_by_label('Prompt', exact=True).fill('Replace the selected area using image 1')
                page.get_by_role('button', name='Generate', exact=True).click()
                expect(page.locator('.main-image')).to_be_visible()
                assert requests[-1]['references'][0] == red
                assert requests[-1]['references'][1] != blue
                assert len(requests[-1]['references']) == 2
                assert 'Edit <image2> as the base' in requests[-1]['prompt']
                page.reload()
                expect(page.get_by_role('button', name='Edit area of image 2')).to_have_text('Area selected')
                page.get_by_role('button', name='Move reference 2 earlier').click()
                expect(page.get_by_role('button', name='Edit area of image 1')).to_have_text('Area selected')
                page.get_by_role('button', name='Edit area of image 1').click()
                expect(page.get_by_role('button', name='Use selection')).to_be_enabled()
                expect(page.get_by_label('Edge softness')).to_have_value('40')
                page.get_by_role('button', name='Cancel', exact=True).click()
                page.get_by_role('button', name='Remove reference 1').click()
                expect(page.get_by_role('button', name='Remove selection')).to_have_count(0)
                page.set_viewport_size({'width':390,'height':844})
                expect(page.get_by_role('button', name='Open sidebar')).to_be_visible()
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                stage, rail = page.locator('.image-stage').bounding_box(), page.locator('.image-rail').bounding_box()
                assert stage['x'] + stage['width'] <= rail['x']
                page.get_by_role('button', name='Open sidebar').click()
                expect(page.get_by_label('Saved sessions')).to_be_visible()
                page.get_by_role('button', name='Edit area of image 1').click()
                expect(page.get_by_label('Paint edit area')).to_be_visible()
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                if args.screenshot:
                    page.screenshot(path=args.screenshot + '.mask.mobile.png', full_page=True)
                page.get_by_role('button', name='Cancel', exact=True).click()
                page.get_by_role('button', name='Close sidebar', exact=True).click()
                if args.screenshot:
                    page.screenshot(path=args.screenshot + '.mobile.png', full_page=True)
                assert not errors, errors
                assert not external, external
                browser.close()
                print('PASS: React uploads, mask painting/undo/restoration/reordering, generation across reload, session persistence, offline browsing, deletion, mobile layout, no external requests')
        finally:
            process.terminate(); process.wait(timeout=10)
            server.shutdown(); server.server_close(); worker.join()


if __name__ == '__main__':
    main()

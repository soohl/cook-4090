"""React API persistence, bounded generation, and file isolation."""

import base64
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from src import image_web
from src.image_history import ImageHistory
from src.image_edit import load_image, png
from PIL import Image


class ImageAPITests(unittest.TestCase):
    def test_area_edit_round_trip_and_mask_survives_restart(self):
        base = Image.new('RGB', (32, 32), 'blue')
        mask = Image.new('L', base.size)
        mask.paste(255, (8, 8, 24, 24))
        encoded = lambda image: base64.b64encode(png(image)).decode()
        refs = [encoded(Image.new('RGB', base.size, 'red')), encoded(base)]
        with tempfile.TemporaryDirectory() as directory, patch.object(image_web, 'store', ImageHistory(directory)), patch.object(image_web, 'jobs', {}):
            async def upstream(method, target, body):
                payload = json.loads(body)
                self.assertEqual(payload['references'][0], refs[0])
                annotated = load_image(base64.b64decode(payload['references'][1]))
                self.assertEqual(annotated.getpixel((0, 0)), (0, 0, 255, 255))
                self.assertNotEqual(annotated.getpixel((12, 12)), (0, 0, 255, 255))
                self.assertEqual(len(payload['references']), 2)
                self.assertIn('Edit <image2> as the base', payload['prompt'])
                self.assertNotIn('edit', payload)
                return httpx.Response(200, json={'image': encoded(Image.new('RGB', base.size, 'green')), 'status': 'done'})
            with patch.object(image_web, 'upstream', side_effect=upstream), TestClient(image_web.app) as client:
                payload = dict(model='image', prompt='Use image 1', size='32x32', steps=25, seed=42,
                               references=refs, edit=dict(base=1, mask=encoded(mask), feather=0))
                invalid = client.post('/api/generate', json=dict(payload, edit=dict(base=2, mask=encoded(mask))))
                self.assertEqual(invalid.status_code, 400)
                self.assertEqual(image_web.jobs, {})
                response = client.post('/api/generate', json=payload)
                self.assertEqual(response.status_code, 202)
                key = response.json()['id']
                deadline = time.monotonic() + 3
                while True:
                    saved = client.get('/api/sessions/' + key).json()
                    if saved['status'] == 'idle':
                        break
                    self.assertNotEqual(saved['status'], 'error', saved)
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.02)
                entry = saved['entries'][0]
                self.assertEqual(entry['prompt'], payload['prompt'])
                self.assertEqual(entry['edit']['base'], 1)
                self.assertEqual(entry['edit']['feather'], 0)
                self.assertEqual(len(entry['references']), 2)
                result = load_image(client.get(entry['image']).content)
                self.assertEqual(result.getpixel((0, 0)), (0, 0, 255, 255))
                self.assertEqual(result.getpixel((12, 12)), (0, 128, 0, 255))
                self.assertNotIn(directory, str(saved))
                image_web.jobs.clear()
                image_web.store = ImageHistory(directory)
                self.assertEqual(client.get('/api/sessions/' + key).json()['entries'], [entry])
                self.assertEqual(load_image(client.get(entry['edit']['mask']).content).convert('L').tobytes(), mask.tobytes())
                self.assertEqual(client.get('/api/sessions/' + key + '/masks/-1').status_code, 404)
                self.assertEqual(client.delete('/api/sessions/' + key).status_code, 204)
                self.assertEqual(client.get(entry['edit']['mask']).status_code, 404)

    def test_generation_persists_references_and_delete_revokes_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(image_web, 'store', ImageHistory(directory)), patch.object(image_web, 'jobs', {}):
            async def upstream(method, target, body):
                self.assertEqual((method, target), ('POST', '/images/generate'))
                return httpx.Response(200, json={'image': base64.b64encode(b'png').decode(), 'status': 'done'})
            with patch.object(image_web, 'upstream', side_effect=upstream), TestClient(image_web.app) as client:
                payload = dict(model='image', prompt='original prompt', size='1024x1024', steps=4, seed=-1,
                               references=[base64.b64encode(b'reference').decode()])
                response = client.post('/api/generate', json=payload)
                self.assertEqual(response.status_code, 202)
                key = response.json()['id']
                deadline = time.monotonic() + 3
                while True:
                    saved = client.get('/api/sessions/' + key).json()
                    if saved['status'] == 'idle':
                        break
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.02)
                entry = saved['entries'][0]
                self.assertNotIn(directory, str(saved))
                self.assertEqual(client.get(entry['image']).content, b'png')
                self.assertEqual(client.get(entry['references'][0]).content, b'reference')
                self.assertGreaterEqual(entry['seed'], 0)
                self.assertEqual(client.get('/api/sessions').json()[0]['id'], key)
                image_web.jobs.clear()  # Equivalent to losing in-memory job state on restart.
                self.assertEqual(client.get('/api/sessions/' + key).json()['entries'], [entry])
                self.assertEqual(client.get('/api/sessions/' + key + '/images/-1').status_code, 404)
                self.assertEqual(client.get('/api/sessions/not-a-session').status_code, 404)
                self.assertEqual(client.delete('/api/sessions/' + key, headers={'sec-fetch-site': 'cross-site'}).status_code, 403)
                self.assertEqual(client.delete('/api/sessions/' + key).status_code, 204)
                self.assertEqual(client.get(entry['image']).status_code, 404)
                self.assertEqual(client.get('/api/sessions').json(), [])

    def test_invalid_requests_and_pending_sessions_cannot_be_deleted(self):
        key = 'a' * 32
        with tempfile.TemporaryDirectory() as directory, patch.object(image_web, 'store', ImageHistory(directory)), patch.object(image_web, 'jobs', {key: {'status': 'generating', 'prompt': 'pending'}}), TestClient(image_web.app) as client:
            self.assertEqual(client.delete('/api/sessions/' + key).status_code, 409)
            payload = dict(session=key, model='image', prompt='p', size='1024x1024', steps=4, seed=42)
            self.assertEqual(client.post('/api/generate', json=payload).status_code, 409)
            self.assertEqual(client.post('/api/generate', json=dict(payload, session='../secret')).status_code, 400)
            self.assertEqual(client.post('/api/generate', json=dict(payload, references=['!!!'])).status_code, 400)
            with patch.dict('os.environ', STACK_MAX_BODY='2'):
                self.assertEqual(client.post('/api/generate', json=payload).status_code, 413)

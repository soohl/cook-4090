"""EXL3 worker isolation and incomplete-download handling, without a GPU."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from src import ROOT


class ExLlamaTests(unittest.TestCase):
    def test_worker_is_offline_loopback_only_and_disposable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, weights, runtime = [root / name for name in ('source', 'weights', 'runtime')]
            source.mkdir()
            weights.mkdir()
            (weights / 'config.json').write_text('{}')
            (weights / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'model.safetensors'}}))
            (weights / 'model.safetensors').write_bytes(b'fixture')
            (source / 'main.py').write_text('''
import json, socket
from pathlib import Path
config = json.loads(Path('config.yml').read_text())
assert config['network']['host'] == '127.0.0.1'
assert config['network']['allowed_origins'] == []
assert config['network']['disable_fetch_requests']
assert [p.name for p in Path(config['model']['model_dir']).iterdir()] == [config['model']['model_name']]
try:
    socket.socket().connect(('127.0.0.1', 9))
except PermissionError:
    pass
else:
    raise AssertionError('outbound connect permitted')
print('isolated worker passed')
''')
            env = dict(os.environ, EXL3_SOURCE=str(source), EXL3_WEIGHTS=str(weights),
                       EXL3_RUNTIME=str(runtime), TMPDIR=str(runtime), CONTEXT='32768', HOST='127.0.0.1')
            # Reuse the test environment. The fake upstream needs no GPU packages.
            import sys
            env['EXL3_PYTHON'] = sys.executable
            command = [str(ROOT / 'run.sh'), 'serve', 'exllamav3']
            result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('isolated worker passed', result.stdout)
            self.assertEqual(list(runtime.iterdir()), [])
            (weights / 'model.safetensors').unlink()
            result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing EXL3 shard', result.stderr)
            result = subprocess.run(command, env=dict(env, HOST='0.0.0.0'), text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('127.0.0.1', result.stderr)

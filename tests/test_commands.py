"""Terminal commands retain results and respect the running inference owner."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from src import ROOT, commands
from src.runtime import Runtime
from src.terminal import Report, capacity, GIB


class CommandTests(unittest.TestCase):
    def test_legacy_ui_has_no_benchmark_tab_or_api(self):
        from src.app import build_ui
        from src.manager import Manager
        from src.registry import load_registry
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            manager = Manager(load_registry(ROOT / 'config/models.json'), directory)
            demo = build_ui(manager)
            self.assertFalse(any(component.get('props', {}).get('label') == 'Benchmarks'
                                 for component in demo.config['components']))
            self.assertFalse(any(dependency.get('api_name') == 'benchmark'
                                 for dependency in demo.config['dependencies']))
            manager.close()

    def test_shell_dispatches_default_and_report_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'run.sh'
            script.write_text((ROOT / 'run.sh').read_text())
            script.chmod(0o755)
            python = root / '.venv/bin/python'
            python.parent.mkdir(parents=True)
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            for option, module in [('', 'inference_service'), ('--verify', 'commands'), ('--benchmark', 'commands')]:
                result = subprocess.run([str(script), *([option] if option else [])], env=dict(os.environ, IMAGE_ENV=str(root/'.venv')), text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(module, result.stdout)
                if option:
                    self.assertIn(option, result.stdout)
                    self.assertNotIn('src.stack', result.stdout)
                else:
                    self.assertLess(result.stdout.index('src.stack\nup'), result.stdout.index('src.inference_service'))
            python.write_text('#!/bin/sh\ncase "$*" in *src.stack*) exit 23;; esac\nprintf "inference started"\n')
            result = subprocess.run([str(script)], env=dict(os.environ, IMAGE_ENV=str(root/'.venv')), text=True, capture_output=True)
            self.assertEqual(result.returncode, 23)
            self.assertNotIn('inference started', result.stdout)

    def test_resource_shortage_fails_without_claiming_readiness(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            report = Report('test', 'verify')
            capacity(report, 'Memory', 8*GIB, 24*GIB)
            self.assertEqual(report.finish(), 1)
        self.assertIn('FAIL', output.getvalue())
        self.assertIn('planning estimate', output.getvalue())

    def test_benchmark_refuses_owned_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = Runtime(directory)
            try:
                with patch.dict(os.environ, UI_RUNTIME=directory), patch.object(commands, 'load_registry', return_value={'test': {}}), patch.object(commands, 'availability', return_value='Installed'), patch.object(commands, 'Manager') as manager, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(commands.benchmark(), 1)
                    manager.assert_not_called()
                self.assertTrue(owner.path.exists())
            finally:
                owner.close()

    def test_benchmark_retains_failed_and_successful_image_reports(self):
        class Manager:
            def __init__(self, registry, runtime):
                self.runtime = runtime
            def stop(self):
                pass
            def close(self):
                pass
        def generate(manager, key, *args):
            if key == 'bad':
                raise RuntimeError('test failure')
            path = manager.runtime / 'image'
            path.mkdir()
            (path/'report.json').write_text(json.dumps({'generation_seconds': 1.25}))
            return '', '', str(path/'report.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = {key: {'kind': 'image'} for key in ('bad', 'good')}
            with patch.dict(os.environ, UI_RUNTIME=str(root/'runtime'), BENCH_REPEATS='1'), patch.object(commands, 'ROOT', root), patch.object(commands, 'load_registry', return_value=registry), patch.object(commands, 'availability', return_value='Installed'), patch.object(commands, 'Manager', Manager), patch.object(commands.images, 'generate', side_effect=generate), patch.object(commands.benchmarks, 'identity', return_value={}), patch.object(commands.benchmarks, 'capture', return_value='test GPU'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(commands.benchmark(), 1)
            report = next((root/'results').glob('*/report.json'))
            result = json.loads(report.read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(len(result['profiles']), 2)
            self.assertTrue((report.parent/'good-1/report.json').exists())
            self.assertEqual(list((root/'runtime').glob('session-*')), [])

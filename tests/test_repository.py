"""Protect the publishable tree from private deployment state."""

from pathlib import Path
import re
import subprocess
import unittest

from src import ROOT
from src.stack import load_env


class RepositoryTests(unittest.TestCase):
    def test_private_files_are_ignored(self):
        for path in ('.env', '.env.backup', 'docs/notes.md', '.local/token',
                     'build/cache', 'models/weights.gguf', 'results/run.json',
                     'private.key', 'private.pem', 'private.pfx', 'identity.db',
                     'identity.db-wal', 'session.sqlite3', 'session.sqlite3-shm'):
            with self.subTest(path=path):
                result = subprocess.run(['git', 'check-ignore', '--no-index', '-q', path], cwd=ROOT)
                self.assertEqual(result.returncode, 0)
        for path in ('config/env.example', 'config/models.json', 'models/.gitkeep'):
            result = subprocess.run(['git', 'check-ignore', '--no-index', '-q', path], cwd=ROOT)
            self.assertEqual(result.returncode, 1, path)
        tracked = subprocess.check_output(
            ['git', 'ls-files', '--cached', '--ignored', '--exclude-standard'], cwd=ROOT)
        self.assertEqual(tracked, b'')

    def test_public_tree_has_no_private_configuration(self):
        names = subprocess.check_output(
            ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT
        ).decode().split('\0')
        private = {}
        if (ROOT / '.env').exists():
            private = {key: value for key, value in load_env(ROOT / '.env').items()
                       if value and key in {'FUNNEL_HOSTNAME', 'CODING_API_KEY',
                                            'POCKET_ID_ENCRYPTION_KEY', 'IMAGE_COOKIE_SECRET',
                                            'OIDC_CLIENT_ID', 'OIDC_CLIENT_SECRET'}}
        for name in set(names):
            path = ROOT / name
            if not name or not path.is_file():
                continue  # Git submodules are directories.
            try:
                content = path.read_text()
            except UnicodeError:
                continue
            with self.subTest(path=name):
                # Report the file and key name, never the matched value.
                self.assertFalse(re.search(r'/(?:Users|home)/[\w.-]+/', content), name)
                self.assertFalse(re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', content), name)
                for key, value in private.items():
                    self.assertFalse(value in content, f'{name}: contains {key}')

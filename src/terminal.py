"""Small, dependency-free terminal reports shared in style by both projects."""

import json
from pathlib import Path
import subprocess
import time

GIB = 1024 ** 3


class Report:
    def __init__(self, project, action):
        self.failed = False
        print(f'\n  {project} / {action}\n  ' + '─' * 66, flush=True)

    def row(self, status, label, detail):
        self.failed |= status == 'FAIL'
        print(f'  {status:<5}  {label:<25} {detail}', flush=True)

    def finish(self):
        print('  ' + '─' * 66, flush=True)
        print('  Result: ' + ('needs attention' if self.failed else 'complete') + '\n', flush=True)
        return int(self.failed)


def capture(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=30).strip()


def check(report, label, function):
    try:
        detail = function()
        report.row('PASS', label, detail or 'available')
        return True
    except (OSError, ImportError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        # Subprocess exception strings may include private command arguments.
        detail = 'command unavailable or failed' if isinstance(exc, subprocess.SubprocessError) else str(exc)
        report.row('FAIL', label, detail)
        return False


def capacity(report, label, available, required):
    report.row('PASS' if available >= required else 'FAIL', label,
               f'{available / GIB:.1f} GiB available; {required / GIB:.1f} GiB required (planning estimate)')


def output_directory(root):
    directory = Path(root) / 'results' / ('benchmark-' + time.strftime('%Y%m%d-%H%M%S'))
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    return directory


def save(directory, report):
    temporary = directory / 'report.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(directory / 'report.json')


def metric(value, unit='s'):
    return 'n/a' if value is None else f'{value:.2f} {unit}'

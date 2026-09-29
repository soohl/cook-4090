"""Terminal setup verification and matched benchmarks for the configured profiles."""

import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import signal

from . import ROOT, benchmarks, images
from .manager import Manager
from .registry import availability, load_registry, profile_env
from .runtime import Runtime
from .stack import load_env, refresh_docker_group
from .terminal import GIB, Report, capacity, capture, check, metric, output_directory, save


def verify():
    report = Report('cook-4090', 'verify')
    report.row('PASS' if sys.platform == 'linux' else 'FAIL', 'Platform', 'Linux required; detected ' + sys.platform)
    check(report, 'Offline worker filter', lambda: ctypes.CDLL('libseccomp.so.2') and 'libseccomp available')
    for package in ('torch', 'diffusers', 'transformers', 'gradio', 'fastapi', 'httpx', 'uvicorn'):
        check(report, package, lambda package=package: importlib.metadata.version(package))
    def cuda():
        import torch
        if not torch.cuda.is_available():
            raise ValueError('PyTorch cannot use CUDA')
        return 'PyTorch CUDA available'
    check(report, 'CUDA runtime', cuda)
    if os.environ.get('IMAGE_QUANTIZATION') == 'int8-convrot' or os.environ.get('IMAGE_ATTENTION') == 'comfy-kitchen':
        def image_kernels():
            from comfy_kitchen import int8_attention_is_available
            if not int8_attention_is_available():
                raise ValueError('Comfy Kitchen CUDA kernels unavailable; run ./run.sh image-setup')
            return 'Comfy Kitchen ' + importlib.metadata.version('comfy-kitchen') + '; CUDA kernels available'
        check(report, 'Image INT8 kernels', image_kernels)
    registry = load_registry(os.environ['UI_REGISTRY'])
    for key, profile in registry.items():
        state = availability(profile)
        required = key in {os.environ['CODING_PROFILE'], 'qwen-image21'}
        report.row('PASS' if state == 'Installed' else 'FAIL' if required else 'SKIP', key, state)
        if state == 'Installed':
            if profile['engine'] == 'exllamav3':
                check(report, 'EXL3 environment', lambda profile=profile: capture([
                    profile_env(profile)['EXL3_PYTHON'], '-c',
                    "import importlib.metadata as m; print('ExLlamaV3 '+m.version('exllamav3')+'; PyTorch '+m.version('torch'))"]))
            for variable in profile['required_env']:
                path = Path(profile_env(profile)[variable])
                if variable.endswith(('SERVER', 'PYTHON')) and not os.access(path, os.X_OK):
                    report.row('FAIL', key, f'{variable} is not executable')
                if path.is_file() and not path.stat().st_size:
                    report.row('FAIL', key, f'{variable} is empty')
    def image_files():
        root = Path(os.environ['IMAGE_WEIGHTS'])
        if not (root / 'model_index.json').is_file():
            raise ValueError('Missing image pipeline metadata')
        indexes = list(root.rglob('*.safetensors.index.json'))
        if not indexes:
            raise ValueError('Missing image weight indexes')
        for index in indexes:
            for name in set(json.loads(index.read_text())['weight_map'].values()):
                file = index.parent / name
                if not file.is_file() or not file.stat().st_size:
                    raise ValueError('An image weight shard is missing or empty')
        return 'indexed weight shards present; checksums not checked'
    check(report, 'Image weight completeness', image_files)
    def gpu():
        rows = capture(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits']).splitlines()
        name, total, free = rows[0].rsplit(',', 2)
        capacity(report, 'GPU capacity', float(total) * 1024**2, float(os.environ['VERIFY_VRAM_GIB']) * GIB)
        report.row('WARN' if float(free)/1024 < float(os.environ['VERIFY_VRAM_GIB']) else 'INFO', 'GPU free now', f'{float(free)/1024:.1f} GiB (running models may occupy memory)')
        return name
    check(report, 'NVIDIA driver / GPU', gpu)
    def ram():
        fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        capacity(report, 'Host RAM', int(fields['MemAvailable'].split()[0])*1024, float(os.environ['VERIFY_RAM_GIB'])*GIB)
    check(report, 'RAM probe', ram)
    capacity(report, 'Free disk', shutil.disk_usage(ROOT).free, float(os.environ['VERIFY_DISK_GIB'])*GIB)
    def credentials():
        path = ROOT / '.env'
        if not path.is_file():
            raise ValueError('Run ./run.sh stack-init and complete enrollment')
        if path.stat().st_mode & 0o077:
            raise ValueError('Run chmod 600 .env')
        values = load_env(path)
        if not all(values.get(key) for key in ('FUNNEL_HOSTNAME', 'CODING_API_KEY', 'POCKET_ID_ENCRYPTION_KEY', 'IMAGE_COOKIE_SECRET', 'OIDC_CLIENT_ID', 'OIDC_CLIENT_SECRET')):
            raise ValueError('Complete the private settings in .env')
        return 'required settings present; values hidden'
    check(report, 'Private configuration', credentials)
    check(report, 'Docker daemon', lambda: capture(['docker', 'info', '--format', '{{.ServerVersion}}']))
    check(report, 'Docker Compose', lambda: capture(['docker', 'compose', 'version', '--short']))
    check(report, 'Stack configuration', lambda: capture([str(ROOT/'run.sh'), 'stack', 'check']) and 'Compose and Caddy valid')
    report.row('INFO', 'Scope', 'static readiness; no model loaded and no memory guarantee')
    return report.finish()


def benchmark():
    report = Report('cook-4090', 'benchmark')
    if not 1 <= int(os.environ['BENCH_REPEATS']) <= 10:
        raise ValueError('BENCH_REPEATS must be between 1 and 10')
    registry = load_registry(os.environ['UI_REGISTRY'])
    selected = [key for key, p in registry.items() if availability(p) == 'Installed']
    for key in registry.keys() - set(selected):
        report.row('SKIP', key, availability(registry[key]))
    if not selected:
        report.row('FAIL', 'Profiles', 'No installed configurations')
        return report.finish()
    # Use the inference runtime lock; never interrupt a running service.
    try:
        runtime = Runtime(os.environ['UI_RUNTIME'])
    except RuntimeError:
        report.row('FAIL', 'GPU ownership', 'Stop your inference service before benchmarking')
        return report.finish()
    try:
        directory = output_directory(ROOT)
    except BaseException:
        runtime.close()
        raise
    manager = Manager(registry, runtime.path)
    summary = {'status': 'running', 'profiles': [], 'settings': {key: value for key, value in os.environ.items() if key.startswith('BENCH_')},
               'image_settings': {'size': os.environ['BENCH_IMAGE_SIZE'], 'steps': int(os.environ['BENCH_IMAGE_STEPS'])},
               'gpu': benchmarks.capture(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'])}
    save(directory, summary)
    try:
        for key in selected:
            report.row('RUN', key, 'fresh engine; model load measured separately')
            try:
                if registry[key]['kind'] == 'chat':
                    paths = []
                    for rows, status, paths in benchmarks.run(manager, [key], os.environ['BENCH_PROMPT'], int(os.environ['BENCH_TOKENS']), int(os.environ['BENCH_REPEATS']), os.environ['BENCH_CACHE'], os.environ['UI_THINKING'] == 'on', int(os.environ['UI_CONTEXT'])):
                        pass
                    record = json.loads(Path(paths[0]).read_text())
                    shutil.copytree(Path(paths[0]).parent, directory/key)
                    if record['status'] != 'complete':
                        raise RuntimeError('Engine benchmark failed; see saved report')
                    for row in rows:
                        report.row('PASS', f'{key} / {row[1]}', f'TTFT {metric(row[5])} · {metric(row[6], "tok/s")} · {row[3]} output tokens')
                else:
                    record = {'profile': key, 'trials': [], 'identity': benchmarks.identity(registry[key])}
                    for trial in range(int(os.environ['BENCH_REPEATS'])):
                        manager.stop()  # Match cold image trials even when serving reuses workers.
                        _, _, path = images.generate(manager, key, os.environ['BENCH_IMAGE_PROMPT'], os.environ['BENCH_IMAGE_SIZE'], int(os.environ['BENCH_IMAGE_STEPS']), int(os.environ['BENCH_SEED']), [])
                        result = json.loads(Path(path).read_text())
                        shutil.copytree(Path(path).parent, directory / f'{key}-{trial+1}')
                        record['trials'].append(result)
                        report.row('PASS', f'{key} / {trial+1}', f'generation {metric(result["generation_seconds"])} · {os.environ["BENCH_IMAGE_SIZE"]} · {os.environ["BENCH_IMAGE_STEPS"]} steps')
                summary['profiles'].append(record)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                report.row('FAIL', key, str(exc).splitlines()[0])
                summary['profiles'].append({'profile': key, 'error': str(exc)})
            save(directory, summary)
        summary['status'] = 'failed' if report.failed else 'complete'
    finally:
        manager.close()
        # Preserve failed worker logs and partial results before removing runtime.
        shutil.copytree(runtime.path, directory/'runtime', dirs_exist_ok=True)
        runtime.close()
        if summary['status'] == 'running':
            summary['status'] = 'interrupted'
        save(directory, summary)
        report.row('INFO', 'Results', str(directory.relative_to(ROOT)))
    return report.finish()


def main():
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        if sys.argv[1] == '--verify':
            refresh_docker_group('src.commands')
        return verify() if sys.argv[1] == '--verify' else benchmark()
    except KeyboardInterrupt:
        print('\n  Interrupted; any started benchmark retains partial results.\n')
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report = Report('cook-4090', sys.argv[1].lstrip('-'))
        report.row('FAIL', 'Setup', str(exc).splitlines()[0])
        return report.finish()


if __name__ == '__main__':
    raise SystemExit(main())

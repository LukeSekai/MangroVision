"""Keep the laptop API and a Cloudflare Quick Tunnel running for user testing."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from dev_processes import own_process_tree, process_options, stop_process_tree
from start_dev import ROOT, STATE_DIR, API_PORT, Controller, WorkspaceLock, ensure_port_available, request_existing_stop

PUBLIC_STATE = STATE_DIR / 'testing.json'


def frontend_origin(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError('Use the HTTPS website origin, without a path or credentials.')
    return value.rstrip('/')


def wait_for_health(url, processes, stopped, timeout=180):
    deadline = time.monotonic() + timeout
    while not stopped.is_set():
        for name, child in processes:
            if child.poll() is not None:
                raise RuntimeError(f'{name} exited (code {child.returncode}).')
        try:
            with urlopen(url, timeout=10) as response:
                if json.load(response).get('status') in ('ok', 'ready'):
                    return
        except (URLError, TimeoutError, ValueError, OSError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f'The API did not become ready at {url}. Check the backend log.')
        stopped.wait(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend-url', type=frontend_origin, help='Stable Vercel production website URL.')
    parser.add_argument('--deploy', action='store_true', help='Update the Vercel frontend to use this tunnel.')
    parser.add_argument('--stop', action='store_true', help='Stop this workspace\'s testing or development server.')
    args = parser.parse_args(argv)
    previous = json.loads(PUBLIC_STATE.read_text(encoding='utf-8')) if PUBLIC_STATE.exists() else {}
    frontend = args.frontend_url or previous.get('frontend_url')
    if not args.stop and not frontend:
        parser.error('Supply --frontend-url https://your-project.vercel.app on the first run.')
    tunnel_binary = shutil.which('cloudflared') or str(STATE_DIR / 'tools' / 'cloudflared.exe')
    if not args.stop and not Path(tunnel_binary).is_file():
        parser.error('cloudflared is missing. See docs/user-testing.md for installation.')
    stopped = threading.Event()
    handlers = {}
    for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            handlers[sig] = signal.signal(sig, lambda *_: stopped.set())
    lock = controller = None
    processes = []
    logs = []
    try:
        lock = WorkspaceLock()
        deadline = time.monotonic() + 25
        requested = False
        while not lock.acquire():
            if stopped.is_set():
                return 0
            if not requested:
                requested = request_existing_stop()
            if time.monotonic() >= deadline:
                raise RuntimeError('The existing workspace launcher did not stop. No additional server was started.')
            stopped.wait(0.1)
        if args.stop:
            print('MangroVision servers stopped.', flush=True)
            return 0
        own_process_tree()
        controller = Controller(stopped)
        ensure_port_available(API_PORT, 'FastAPI', stopped)
        if sys.platform == 'win32':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        env = os.environ.copy()
        env.update({
            'PYTHONPATH': os.pathsep.join([str(ROOT.parent), str(ROOT), env.get('PYTHONPATH', '')]),
            'PYTHONIOENCODING': 'utf-8', 'COOKIE_SECURE': 'true', 'COOKIE_DOMAIN': '',
            'TRUSTED_ORIGINS': f'{frontend},http://localhost:5173,http://127.0.0.1:5173',
            'MANGROVISION_CORS_ORIGINS': frontend,
        })
        backend_log = (STATE_DIR / 'testing-backend.log').open('w', encoding='utf-8')
        logs.append(backend_log)
        backend = subprocess.Popen(
            [sys.executable, '-X', 'utf8', '-m', 'uvicorn', 'api.main:app', '--host', '127.0.0.1',
             '--port', str(API_PORT), '--workers', '1'],
            cwd=ROOT, env=env, stdout=backend_log, stderr=subprocess.STDOUT, **process_options(),
        )
        processes.append(('FastAPI', backend))
        print('Starting laptop API (one worker, no reload)...', flush=True)
        wait_for_health(f'http://127.0.0.1:{API_PORT}/api/health/ready', processes, stopped)
        if stopped.is_set():
            return 0
        tunnel_log = (STATE_DIR / 'testing-tunnel.log').open('w', encoding='utf-8')
        logs.append(tunnel_log)
        tunnel = subprocess.Popen(
            [tunnel_binary, 'tunnel', '--url', f'http://127.0.0.1:{API_PORT}', '--no-autoupdate', '--protocol', 'http2'],
            cwd=STATE_DIR, stdout=tunnel_log, stderr=subprocess.STDOUT, **process_options(),
        )
        processes.append(('Cloudflare tunnel', tunnel))
        deadline = time.monotonic() + 90
        tunnel_url = None
        while not stopped.is_set():
            if tunnel.poll() is not None:
                raise RuntimeError('Cloudflare tunnel exited. Check .dev-server/testing-tunnel.log.')
            content = (STATE_DIR / 'testing-tunnel.log').read_text(encoding='utf-8', errors='replace')
            match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', content)
            if match:
                tunnel_url = match.group(0)
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Cloudflare did not provide a tunnel URL within 90 seconds.')
            stopped.wait(0.5)
        if stopped.is_set():
            return 0
        wait_for_health(f'{tunnel_url}/api/health/live', processes, stopped, timeout=90)
        if stopped.is_set():
            return 0
        PUBLIC_STATE.write_text(json.dumps({
            'frontend_url': frontend, 'backend_url': tunnel_url, 'pid': os.getpid(),
        }, indent=2), encoding='utf-8')
        print(f'\nLaptop API is online: {tunnel_url}\nWebsite: {frontend}', flush=True)
        if args.deploy:
            deployer = subprocess.Popen([sys.executable, '-X', 'utf8', str(ROOT / 'deploy_testing.py')], cwd=ROOT.parent, **process_options())
            processes.append(('Vercel deployment', deployer))
            while deployer.poll() is None and not stopped.wait(0.2):
                if backend.poll() is not None or tunnel.poll() is not None:
                    raise RuntimeError('The laptop API or tunnel stopped during deployment.')
            if stopped.is_set():
                return 0
            processes.remove(('Vercel deployment', deployer))
            if deployer.returncode:
                print('Deployment failed. The API stays running; retry python MangroVision_New/deploy_testing.py.', flush=True)
        else:
            print('Update the website connection: python MangroVision_New/deploy_testing.py', flush=True)
        print('Keep this terminal open. Windows sleep is held off while this launcher runs.\nPress Ctrl+C to stop the API and tunnel.', flush=True)
        while not stopped.wait(0.5):
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f'{name} stopped; shutting down the testing stack.')
        return 0
    except (OSError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr, flush=True)
        return 1
    finally:
        for _, process in reversed(processes):
            stop_process_tree(process)
        for log in logs:
            log.close()
        if controller:
            controller.close()
        if lock:
            lock.close()
        if sys.platform == 'win32':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())

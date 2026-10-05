"""Connect the Vercel frontend to the currently running laptop tunnel."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
CLIENT = ROOT / 'client'
STATE = ROOT / '.dev-server' / 'testing.json'


def main():
    if not STATE.is_file():
        raise RuntimeError('Start start_testing.py first; no running tunnel is recorded.')
    state = json.loads(STATE.read_text(encoding='utf-8'))
    backend = state.get('backend_url', '')
    if not re.fullmatch(r'https://[a-z0-9-]+\.trycloudflare\.com', backend):
        raise RuntimeError('The recorded backend URL is not a Cloudflare Quick Tunnel.')
    with urlopen(f'{backend}/api/health/ready', timeout=30) as response:
        if json.load(response).get('status') != 'ready':
            raise RuntimeError('The laptop API, database, and storage must be ready before deployment.')
    local_cli = ROOT.parent / 'tmp' / 'vercel-cli' / 'node_modules' / 'vercel' / 'dist' / 'vc.js'
    if local_cli.is_file() and shutil.which('node'):
        command = [shutil.which('node'), str(local_cli)]
    elif shutil.which('vercel'):
        command = [shutil.which('vercel')]
    else:
        raise RuntimeError('Install the Vercel CLI as described in docs/user-testing.md.')
    if not (CLIENT / '.vercel' / 'project.json').is_file():
        raise RuntimeError('Link the frontend to your Vercel project first (see docs/user-testing.md).')
    env = os.environ.copy()
    env['MANGROVISION_BACKEND_URL'] = backend
    env['VERCEL_TELEMETRY_DISABLED'] = '1'
    # Public routing settings only. Database/storage credentials stay in the laptop .env.
    for name, value in (('MANGROVISION_BACKEND_URL', backend), ('VITE_API_BASE', ''), ('VITE_TILE_SERVER', '/tiles')):
        subprocess.run(command + ['env', 'add', name, 'production', '--value', value, '--force', '--yes', '--no-sensitive'],
                       cwd=CLIENT, env=env, check=True)
    # Match the project's Git root directory, with a strict upload allowlist.
    metadata = ROOT.parent / '.vercel' / 'project.json'
    metadata.parent.mkdir(exist_ok=True)
    shutil.copyfile(CLIENT / '.vercel' / 'project.json', metadata)
    subprocess.run(command + ['deploy', '--prod', '--yes', '--force'],
                   cwd=ROOT.parent, env=env, check=True)
    print(f"\nShare this website with the expert: {state['frontend_url']}")
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Deployment failed: {error}', file=sys.stderr)
        raise SystemExit(1)

"""Run one supervised MangroVision development stack.

Running this command again restarts this workspace's existing managed stack.
Use python start_dev.py --stop to stop it from another terminal.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from dev_processes import own_process_tree, process_options, stop_process_tree

ROOT = Path(__file__).resolve().parent
CLIENT_DIR = ROOT / "client"
API_DIR = ROOT / "api"
STATE_DIR = ROOT / ".dev-server"
API_PORT = 8000
FRONTEND_PORT = 5173


class WorkspaceLock:
    """An OS lock, automatically released after a crash; never trust a PID file."""

    def __init__(self):
        STATE_DIR.mkdir(exist_ok=True)
        self.file = (STATE_DIR / "launcher.lock").open("a+b")
        self.locked = False
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()

    def acquire(self):
        self.file.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        self.locked = True
        return True

    def close(self):
        self.file.close()  # Closing releases the lock on both platforms.


def request_existing_stop():
    """Stop only a launcher with this workspace's private control capability."""
    try:
        state = json.loads((STATE_DIR / "control.json").read_text(encoding="utf-8"))
        if state["workspace"] != str(ROOT):
            return False
        with socket.create_connection(("127.0.0.1", int(state["port"])), timeout=1) as connection:
            connection.sendall((json.dumps({"token": state["token"], "command": "stop"}) + "\n").encode())
            with connection.makefile("rb") as stream:
                reply = json.loads(stream.readline(2048))
            return isinstance(reply, dict) and reply.get("status") == "stopping" and reply.get("workspace") == str(ROOT)
    except (OSError, ValueError, KeyError, TypeError):
        return False


class Controller:
    def __init__(self, stopped):
        self.stopped = stopped
        self.closed = threading.Event()
        self.token = secrets.token_hex(32)
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(4)
        self.socket.settimeout(0.25)
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()
        state = {"workspace": str(ROOT), "pid": os.getpid(),
                 "port": self.socket.getsockname()[1], "token": self.token}
        temporary = STATE_DIR / "control.tmp"
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(STATE_DIR / "control.json")

    def serve(self):
        while not self.closed.is_set():
            try:
                connection, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                try:
                    connection.settimeout(1)
                    with connection.makefile("rb") as stream:
                        request = json.loads(stream.readline(2048))
                    if not isinstance(request, dict):
                        continue
                    token = request.get("token")
                    if not isinstance(token, str) or not hmac.compare_digest(token, self.token):
                        continue
                    if request.get("command") == "stop":
                        self.stopped.set()
                        connection.sendall((json.dumps({"status": "stopping", "workspace": str(ROOT)}) + "\n").encode())
                except (OSError, ValueError, TypeError):
                    continue

    def close(self):
        self.closed.set()
        self.socket.close()
        self.thread.join(timeout=2)
        (STATE_DIR / "control.json").unlink(missing_ok=True)


def ensure_port_available(port, service, stopped):
    """Allow previous sockets to settle, but never kill an unrelated port owner."""
    deadline = time.monotonic() + 3
    while not stopped.is_set():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                if sys.platform == "win32":
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("0.0.0.0", port))
            return
        except OSError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{service} port {port} is occupied or blocked by a process outside this launcher.\n"
                    f"  Get-NetTCPConnection -State Listen -LocalPort {port} | "
                    "Select-Object LocalPort, OwningProcess\n"
                    f"Windows reported: {error}"
                ) from error
            stopped.wait(0.1)


def server_commands():
    node = shutil.which("node")
    vite = CLIENT_DIR / "node_modules" / "vite" / "bin" / "vite.js"
    if not node or not vite.is_file():
        raise RuntimeError("Node/Vite is unavailable. Install the client dependencies with npm install.")
    return [
        ([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0",
          "--port", str(API_PORT), "--reload", "--reload-dir", str(API_DIR),
          "--reload-dir", str(ROOT.parent / "canopy_detection")], ROOT),
        ([node, str(vite), "--port", str(FRONTEND_PORT), "--strictPort"], CLIENT_DIR),
    ]


def wait_for_server(process, port, processes, stopped, timeout=120):
    deadline = time.monotonic() + timeout
    while not stopped.is_set():
        for name, child in processes:
            code = child.poll()
            if code is not None:
                raise RuntimeError(f"{name} exited during startup (code {code}).")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Server did not become ready on port {port} within {timeout}s.")
            stopped.wait(0.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop", action="store_true", help="Stop this workspace's running development stack.")
    args = parser.parse_args(argv)
    stopped = threading.Event()
    previous_signals = {}
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            previous_signals[sig] = signal.signal(sig, lambda *_: stopped.set())
    lock = None
    controller = None
    processes = []
    try:
        lock = WorkspaceLock()
        deadline = time.monotonic() + 25
        requested_stop = False
        while not lock.acquire():
            if stopped.is_set():
                return 0
            if not requested_stop and request_existing_stop():
                requested_stop = True
                print("Stopping existing MangroVision server..." if args.stop else
                      "Restarting this workspace's existing MangroVision server...", flush=True)
            if time.monotonic() >= deadline:
                raise RuntimeError("The existing launcher did not shut down within 25 seconds. No additional servers were started.")
            stopped.wait(0.1)
        if args.stop:
            (STATE_DIR / "control.json").unlink(missing_ok=True)
            print("MangroVision development servers are stopped.", flush=True)
            return 0
        if stopped.is_set():
            return 0

        # Kernel ownership is established before ANY child process is created.
        own_process_tree()
        controller = Controller(stopped)
        ensure_port_available(API_PORT, "FastAPI", stopped)
        ensure_port_available(FRONTEND_PORT, "Vite", stopped)
        commands = server_commands()
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT.parent), str(ROOT)] + ([child_env["PYTHONPATH"]] if child_env.get("PYTHONPATH") else [])
        )
        print("Starting MangroVision development servers...", flush=True)
        for (command, directory), name, port in zip(commands, ("FastAPI", "Vite"), (API_PORT, FRONTEND_PORT)):
            if stopped.is_set():
                return 0
            process = subprocess.Popen(command, cwd=str(directory), env=child_env, **process_options())
            processes.append((name, process))
            wait_for_server(process, port, processes, stopped)
        if stopped.is_set():
            return 0
        print(
            f"\nMangroVision is ready.\n"
            f"  Frontend: http://localhost:{FRONTEND_PORT}\n"
            f"  API docs: http://localhost:{API_PORT}/api/docs\n"
            "Press Ctrl+C to stop both servers. Running start_dev.py again restarts this stack.\n",
            flush=True,
        )
        while not stopped.wait(0.2):
            for name, process in processes:
                code = process.poll()
                if code is not None:
                    raise RuntimeError(f"{name} stopped (code {code}); shutting down both servers.")
        return 0
    except (OSError, RuntimeError) as error:
        print(f"\nERROR: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        # Repeated Ctrl+C sets the event instead of interrupting cleanup.
        if processes:
            print("\nShutting down both MangroVision servers...", flush=True)
        for _, process in reversed(processes):
            try:
                stop_process_tree(process)
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"Cleanup: {error}", file=sys.stderr)
        if controller is not None:
            controller.close()
        if lock is not None:
            lock.close()
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())

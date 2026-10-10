import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from .display_mode import DisplaySnapshot, monitor_api

HEARTBEAT_SECONDS = 1.0
RECOVERY_TIMEOUT = 10.0


class ModeWatchdog:
    def __init__(self, snapshot: DisplaySnapshot) -> None:
        self.stop = threading.Event()
        command = [
            sys.executable,
            "-m",
            "iw3.desktop.display_mode_guard",
            "--parent",
            str(os.getpid()),
            "--created",
            str(psutil.Process().create_time()),
            "--snapshot",
            json.dumps(snapshot.to_dict()),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32",
        )
        ready: queue.Queue[str] = queue.Queue()

        def read_ready() -> None:
            assert self.process.stdout is not None
            try:
                ready.put(self.process.stdout.readline().strip())
            finally:
                self.process.stdout.close()

        threading.Thread(target=read_ready, daemon=True).start()
        try:
            if ready.get(timeout=8.0) != "ready":
                raise RuntimeError("Display mode watchdog could not initialize")
        except (queue.Empty, RuntimeError) as exc:
            self.process.kill()
            self.process.wait()
            if self.process.stdin is not None:
                self.process.stdin.close()
            raise RuntimeError("Display mode watchdog did not become ready; monitor mode was not changed") from exc
        self.heartbeat = threading.Thread(target=self._send_heartbeats, daemon=True)
        self.heartbeat.start()

    def _send_heartbeats(self) -> None:
        assert self.process.stdin is not None
        try:
            while not self.stop.wait(HEARTBEAT_SECONDS):
                self.process.stdin.write("ping\n")
                self.process.stdin.flush()
        except (OSError, ValueError):
            print("Display mode watchdog connection was lost", file=sys.stderr)

    def release(self, *, restored: bool) -> None:
        self.stop.set()
        self.heartbeat.join(timeout=2.0)
        pipe = self.process.stdin
        if pipe is not None and not pipe.closed:
            try:
                if restored:
                    pipe.write("done\n")
                    pipe.flush()
                pipe.close()
            except (OSError, ValueError):
                pass
        if restored:
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def parent_alive(pid: int, created: float) -> bool:
    try:
        parent = psutil.Process(pid)
        return parent.is_running() and abs(parent.create_time() - created) < 0.01
    except psutil.Error:
        return False


def watch_parent(pid: int, created: float, snapshot: DisplaySnapshot) -> int:
    monitor_api()
    messages: queue.Queue[tuple[str, float]] = queue.Queue()

    def receive() -> None:
        try:
            for line in sys.stdin:
                messages.put((line.strip(), time.monotonic()))
        finally:
            messages.put(("eof", time.monotonic()))

    threading.Thread(target=receive, daemon=True).start()
    print("ready", flush=True)
    last_ping = time.monotonic()
    while True:
        try:
            message, received = messages.get(timeout=HEARTBEAT_SECONDS)
        except queue.Empty:
            message, received = "", last_ping
        if message == "done":
            return 0
        if message == "ping":
            last_ping = received
        if message == "eof" or not parent_alive(pid, created) or time.monotonic() - last_ping >= RECOVERY_TIMEOUT:
            break

    for attempt in range(3):
        try:
            snapshot.restore()
            print(f"Local Viewer watchdog restored {snapshot.monitor_name}", file=sys.stderr)
            return 0
        except Exception as exc:
            print(f"Local Viewer watchdog restoration failed: {exc}", file=sys.stderr)
            if attempt < 2:
                time.sleep(HEARTBEAT_SECONDS)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Viewer display mode recovery watchdog")
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--created", type=float, required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    return watch_parent(args.parent, args.created, DisplaySnapshot.from_dict(json.loads(args.snapshot)))


if __name__ == "__main__":
    raise SystemExit(main())

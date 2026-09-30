"""Bounded child jobs, including descendant cleanup on interrupt or timeout."""

import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def study_lock(root):
    with (Path(root) / ".controller.lock").open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another controller is using this study") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def stop_process(process):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        # Also kill descendants that ignored SIGTERM after the parent exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_job(command, *, cwd, log, timeout, prompt=None):
    if timeout <= 0:
        raise TimeoutError("Study time budget exhausted")
    env = os.environ.copy()
    env.pop("AUTOVLA_API_KEY", None)
    env["PYTHONUNBUFFERED"] = "1"
    if os.name != "nt":
        env.setdefault("MUJOCO_GL", "egl")
    with Path(log).open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if prompt is not None else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            start_new_session=os.name != "nt",
        )
        started = time.monotonic()
        try:
            if prompt is not None:
                process.stdin.write(prompt)
                process.stdin.close()
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f"Job exceeded {timeout:.0f}s; see {log}")
                try:
                    code = process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(f"Job running; log: {log}", flush=True)
            if code:
                raise RuntimeError(f"Job exited with {code}; see {log}")
        except BaseException:
            stop_process(process)
            raise

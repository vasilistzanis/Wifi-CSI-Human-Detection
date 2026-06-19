#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralised run-log helper for every runnable script in this project.

Each script calls `setup_run_log(__file__)` exactly once at startup.  That
opens a fresh log file under  <project root>/logs/  whose name encodes both
the originating script and the run timestamp, then tees every stdout/stderr
write into it so the file mirrors what the user sees in the terminal.

Filename format:
    <script>_<YYYY-MM-DD_HH-MM-SS>_pid<PID>.log

Example:
    logs/benchmark_latency_2026-06-19_23-58-12_pid12480.log

The PID suffix keeps the filename unique when two scripts start in the same
second (which is otherwise the timestamp resolution).
"""

import os
import sys
import atexit
import datetime
from pathlib import Path


_INSTALLED = False   # idempotency guard — calling twice in one process is a no-op


def setup_run_log(script_path: "str | Path", logs_dir: "str | Path | None" = None) -> Path:
    """Tee stdout + stderr to a timestamped log file.

    Args:
        script_path: usually ``__file__`` from the calling script.
        logs_dir:    target folder; defaults to ``<this file's dir>/logs``
                     so the log path is stable regardless of the caller's CWD.

    Returns:
        The Path of the newly opened log file.  Calling again in the same
        process is a no-op and returns the original path.
    """
    global _INSTALLED
    if _INSTALLED:
        return Path(_INSTALLED)   # type: ignore[arg-type]

    script_name = Path(script_path).stem

    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"{script_name}_{ts}_pid{os.getpid()}.log"

    # Line-buffered so the file stays usable while the script is still running.
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    class _Tee:
        """Forward writes to both the original stream and the log file.

        Tolerates either side being closed without crashing the script —
        important for atexit / signal-driven shutdowns (PyQt, Ctrl+C, etc.).
        """
        def __init__(self, console, log):
            self._console = console
            self._log = log

        def write(self, data):
            try:
                self._console.write(data)
            except Exception:
                pass
            try:
                self._log.write(data)
            except (ValueError, OSError):
                pass

        def flush(self):
            try:
                self._console.flush()
            except Exception:
                pass
            try:
                self._log.flush()
            except (ValueError, OSError):
                pass

        def isatty(self):
            return getattr(self._console, "isatty", lambda: False)()

        def fileno(self):
            return self._console.fileno()

        def __getattr__(self, name):
            return getattr(self._console, name)

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    started_at = datetime.datetime.now()
    print("=" * 72)
    print(f"  LOG FILE  : {log_path}")
    print(f"  Script    : {script_name}")
    print(f"  Started   : {started_at.isoformat(timespec='seconds')}")
    print(f"  CWD       : {Path.cwd()}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  PID       : {os.getpid()}")
    print(f"  Args      : {' '.join(sys.argv)}")
    print("=" * 72)
    print()

    def _close():
        try:
            ended_at = datetime.datetime.now()
            duration = ended_at - started_at
            print()
            print("=" * 72)
            print(f"  Finished  : {ended_at.isoformat(timespec='seconds')}")
            print(f"  Duration  : {duration}")
            print("=" * 72)
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    atexit.register(_close)
    _INSTALLED = str(log_path)
    return log_path

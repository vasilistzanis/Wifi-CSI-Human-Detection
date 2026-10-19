"""Compatibility launcher and inference exports for live prediction."""

from apps.live_predict import *
from apps.live_predict import main as _main


if __name__ == "__main__":
    from logging_setup import setup_run_log

    setup_run_log(__file__)
    raise SystemExit(_main())

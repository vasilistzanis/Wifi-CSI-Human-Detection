"""Compatibility launcher for the desktop dashboard."""

from apps.live_dashboard import *
from apps.live_dashboard import main as _main


if __name__ == "__main__":
    from logging_setup import setup_run_log

    setup_run_log(__file__)
    _main()

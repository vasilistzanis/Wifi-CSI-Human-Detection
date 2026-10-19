"""Compatibility launcher for the live sensing viewer."""

from apps.live_sensing_1 import *
from apps.live_sensing_1 import main as _main


if __name__ == "__main__":
    from logging_setup import setup_run_log

    setup_run_log(__file__)
    _main()

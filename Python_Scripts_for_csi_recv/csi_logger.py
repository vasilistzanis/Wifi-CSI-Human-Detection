"""Compatibility launcher for the serial CSI logger."""

from apps.csi_logger import *
from apps.csi_logger import main as _main


if __name__ == "__main__":
    from logging_setup import setup_run_log

    setup_run_log(__file__)
    raise SystemExit(_main())

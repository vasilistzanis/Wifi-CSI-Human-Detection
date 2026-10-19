"""Compatibility launcher for the dual-view CSI viewer."""

from apps.live_csi_dual_view import *
from apps.live_csi_dual_view import main as _main


if __name__ == "__main__":
    from logging_setup import setup_run_log

    setup_run_log(__file__)
    _main()

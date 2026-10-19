"""Compatibility launcher for the experiment runner."""

from experiments.experiment_runner import *
from experiments.experiment_runner import main as _main


if __name__ == "__main__":
    from logging_setup import setup_run_log

    setup_run_log(__file__)
    raise SystemExit(_main())

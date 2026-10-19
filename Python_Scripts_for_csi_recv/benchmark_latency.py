"""Compatibility launcher for the latency benchmark."""

from experiments.benchmark_latency import *
from experiments.benchmark_latency import main as _main


if __name__ == "__main__":
    from logging_setup import setup_run_log

    setup_run_log(__file__)
    _main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import time
from pathlib import Path

import serial

DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 2_000_000
DEFAULT_IDLE_SLEEP = 0.001
DEFAULT_FLUSH_INTERVAL = 0.5
DEFAULT_STATUS_INTERVAL = 0.25
DEFAULT_SERIAL_BUFFER_SIZE = 2_000_000

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="High-speed ESP32 CSI logger")
    parser.add_argument("-p", "--port", default=DEFAULT_PORT, help="Serial port, e.g. COM6")
    parser.add_argument("-b", "--baud", type=int, default=DEFAULT_BAUD, help="Baud rate")
    parser.add_argument("-l", "--label", help="Dataset label. If omitted, you will be prompted.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="datasets",
        help="Output directory. Relative paths are resolved from this script.",
    )
    parser.add_argument(
        "--idle-sleep",
        type=float,
        default=DEFAULT_IDLE_SLEEP,
        help="Sleep time when the serial buffer is empty.",
    )
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=DEFAULT_FLUSH_INTERVAL,
        help="Seconds between file flushes.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=DEFAULT_STATUS_INTERVAL,
        help="Seconds between progress updates.",
    )
    parser.add_argument(
        "--serial-buffer-size",
        type=int,
        default=DEFAULT_SERIAL_BUFFER_SIZE,
        help="Windows RX buffer size in bytes.",
    )
    return parser.parse_args()


def sanitize_label(raw_label: str) -> str:
    label = re.sub(r'[<>:"/\\|?*\s]+', "_", raw_label.strip())
    label = re.sub(r"_+", "_", label).strip("._")
    return label


def resolve_output_dir(output_dir_arg: str) -> Path:
    output_dir = Path(output_dir_arg)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    return output_dir


def safe_set_buffer_size(ser: serial.Serial, rx_size: int) -> None:
    if os.name != "nt" or not hasattr(ser, "set_buffer_size"):
        return

    try:
        ser.set_buffer_size(rx_size=rx_size)
    except Exception:
        pass


def prompt_for_label() -> str:
    return input("Enter capture label (e.g. walk_1, fall_3, empty): ")


def main() -> int:
    args = parse_args()

    raw_label = args.label if args.label else prompt_for_label()
    label = sanitize_label(raw_label)
    if not label:
        print("No valid label was provided. Exiting.")
        return 1

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{label}_{int(time.time())}.txt"
    bytes_written = 0
    capture_started = False
    start_time = time.monotonic()
    last_flush = start_time
    last_status = start_time
    ser = None

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
        safe_set_buffer_size(ser, args.serial_buffer_size)

        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        print("\n" + "=" * 48)
        print("ESP32-C6 RADAR - HIGH SPEED LOGGER")
        print("=" * 48)
        print(f"Port   : {args.port}")
        print(f"Baud   : {args.baud}")
        print(f"Label  : {label}")
        print(f"Output : {output_path}")
        print("Press Ctrl+C to stop.\n")

        with open(output_path, "wb") as handle:
            capture_started = True

            while True:
                waiting = ser.in_waiting
                if waiting <= 0:
                    time.sleep(args.idle_sleep)
                    continue

                chunk = ser.read(waiting)
                if not chunk:
                    continue

                handle.write(chunk)
                bytes_written += len(chunk)

                now = time.monotonic()

                if now - last_flush >= args.flush_interval:
                    handle.flush()
                    last_flush = now

                if now - last_status >= args.status_interval:
                    elapsed = max(now - start_time, 1e-6)
                    kb_written = bytes_written / 1024.0
                    avg_rate = kb_written / elapsed
                    print(
                        f"\rCaptured: {kb_written:10.1f} KB | Avg rate: {avg_rate:8.1f} KB/s",
                        end="",
                        flush=True,
                    )
                    last_status = now

    except serial.SerialException as exc:
        print(f"\nSerial error on {args.port}: {exc}")
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        if ser is not None and ser.is_open:
            ser.close()

    if capture_started:
        elapsed = max(time.monotonic() - start_time, 1e-6)
        kb_written = bytes_written / 1024.0
        avg_rate = kb_written / elapsed
        print("\n")
        print(f"Capture finished: {kb_written:.1f} KB written")
        print(f"Elapsed time    : {elapsed:.2f} s")
        print(f"Average rate    : {avg_rate:.1f} KB/s")
        print(f"Saved to        : {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-speed ESP32 CSI logger with improved validation and error handling.
"""

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import serial
from serial.tools import list_ports


def configure_console_output() -> None:
    """Avoid UnicodeEncodeError on legacy Windows console encodings."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


configure_console_output()


# Cross-platform defaults
DEFAULT_PORT = "COM6" if os.name == "nt" else "/dev/ttyUSB0"

# Mismatch causes garbled output and zero valid frames.
DEFAULT_BAUD = 2_000_000
DEFAULT_IDLE_SLEEP = 0.001
DEFAULT_FLUSH_INTERVAL = 0.5
DEFAULT_STATUS_INTERVAL = 0.25
DEFAULT_SERIAL_BUFFER_SIZE = 2_000_000
MAX_FILE_SIZE_MB = 500  # Safety limit to prevent filling disk

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
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=MAX_FILE_SIZE_MB,
        help="Maximum file size in MB (safety limit).",
    )
    parser.add_argument(
        "-w", "--wait",
        type=int,
        default=5,
        help="Countdown in seconds before recording starts. Set to 0 to disable. (default: 5)",
    )
    return parser.parse_args()


def sanitize_label(raw_label: str) -> str:
    """Remove unsafe characters from label."""
    label = re.sub(r'[<>:"/\\|?*\s]+', "_", raw_label.strip())
    label = re.sub(r"_+", "_", label).strip("._")
    return label


def resolve_output_dir(output_dir_arg: str) -> Path:
    """Resolve output directory path."""
    output_dir = Path(output_dir_arg)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    return output_dir


def list_available_ports() -> list[str]:
    """Return list of available serial port names."""
    return [p.device for p in list_ports.comports()]


def validate_port(port: str) -> bool:
    """Check if serial port exists and is accessible."""
    available_ports = list_available_ports()
    return port in available_ports


def safe_set_buffer_size(ser: serial.Serial, rx_size: int) -> None:
    """Set serial buffer size (Windows only)."""
    if os.name != "nt" or not hasattr(ser, "set_buffer_size"):
        return

    try:
        ser.set_buffer_size(rx_size=rx_size)
    except Exception:
        pass


def prompt_for_label() -> str:
    """Prompt user for capture label."""
    return input("Enter capture label (e.g. walk_1, fall_3, empty): ")


def main() -> int:
    args = parse_args()

    # ── Port Validation ───────────────────────────────────────────────────
    if not validate_port(args.port):
        print(f"❌ Port '{args.port}' not found or not accessible.")
        available = list_available_ports()
        if available:
            print("\n📡 Available ports:")
            for port in available:
                print(f"   {port}")
            print("\nUse -p <PORT> to specify a different port.")
        else:
            print("⚠️  No serial ports detected. Check your USB connection.")
        return 1

    # ── Label Validation ──────────────────────────────────────────────────
    raw_label = args.label if args.label else prompt_for_label()
    label = sanitize_label(raw_label)
    if not label:
        print("❌ No valid label was provided. Exiting.")
        return 1

    # ── Output Directory Setup ────────────────────────────────────────────
    output_dir = resolve_output_dir(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as exc:
        print(f"❌ Cannot create output directory {output_dir}: {exc}")
        return 1

    output_path = output_dir / f"{label}_{int(time.time())}.txt"
    max_bytes = args.max_size_mb * 1024 * 1024
    bytes_written = 0
    capture_started = False
    start_time = time.monotonic()
    last_flush = start_time
    last_status = start_time
    ser = None

    try:
        # ── Open Serial Port ──────────────────────────────────────────────
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
        print(f"Max    : {args.max_size_mb} MB")
        print("Press Ctrl+C to stop.\n")

        # ── Countdown Timer ───────────────────────────────────────────────
        if args.wait > 0:
            print(f"⏳ Έναρξη σε {args.wait} δευτερόλεπτα! (Πάρε θέση...)")
            for i in range(args.wait, 0, -1):
                print(f"   {i}...")
                time.sleep(1)
            print("▶️  ΠΑΜΕ! (Η καταγραφή ξεκίνησε)\n")
        
        # Clear buffer immediately before the file opens so we don't log the movement of pressing enter
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        # Also reset tracking timers so they don't count the wait time
        start_time = time.monotonic()
        last_flush = start_time
        last_status = start_time

        with open(output_path, "wb", buffering=1024*1024) as handle:
            capture_started = True

            while True:
                # Safety net: catches any edge case where bytes_written
                # already equals max_bytes at loop entry
                if bytes_written >= max_bytes:
                    print(f"\n⚠️  Reached maximum file size ({args.max_size_mb} MB)")
                    print("   Stopping capture.")
                    break

                waiting = ser.in_waiting
                if waiting <= 0:
                    time.sleep(args.idle_sleep)
                    continue

                chunk = ser.read(waiting)
                if not chunk:
                    continue

                # Prevent disk overflow by checking size BEFORE writing the chunk

                if bytes_written + len(chunk) > max_bytes:
                    print(f"\n⚠️  Reached maximum file size ({args.max_size_mb} MB)")
                    print("   Stopping capture to prevent disk overflow.")
                    break

                handle.write(chunk)
                bytes_written += len(chunk)

                now = time.monotonic()

                # ── Periodic File Flush ───────────────────────────────────
                if now - last_flush >= args.flush_interval:
                    handle.flush()
                    last_flush = now

                # ── Status Updates ────────────────────────────────────────
                if now - last_status >= args.status_interval:
                    elapsed = max(now - start_time, 1e-6)
                    kb_written = bytes_written / 1024.0
                    avg_rate = kb_written / elapsed
                    percent_full = (bytes_written / max_bytes) * 100
                    print(
                        f"\rCaptured: {kb_written:10.1f} KB "
                        f"({percent_full:5.1f}%) | "
                        f"Avg rate: {avg_rate:8.1f} KB/s",
                        end="",
                        flush=True,
                    )
                    last_status = now

    except serial.SerialException as exc:
        print(f"\n❌ Serial error on {args.port}: {exc}")
        print("   Check that the device is connected and not in use.")
        return 1
    except PermissionError as exc:
        print(f"\n❌ Permission denied: {exc}")
        print(f"   Cannot write to {output_path}")
        return 1
    except KeyboardInterrupt:
        print("\n\n⏹️  Capture interrupted by user (Ctrl+C)")
    except Exception as exc:
        print(f"\n❌ Unexpected error: {exc}")
        return 1
    finally:
        if ser is not None and ser.is_open:
            ser.close()

    # ── Final Statistics ──────────────────────────────────────────────────
    if capture_started:
        elapsed = max(time.monotonic() - start_time, 1e-6)
        kb_written = bytes_written / 1024.0
        avg_rate = kb_written / elapsed
        print("\n")
        print("=" * 48)
        print(f"✅ Capture finished")
        print("=" * 48)
        print(f"Total written   : {kb_written:.1f} KB")
        print(f"Elapsed time    : {elapsed:.2f} s")
        print(f"Average rate    : {avg_rate:.1f} KB/s")
        print(f"File location   : {output_path}")
        
        # Verify file was written
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"File size       : {output_path.stat().st_size / 1024:.1f} KB")
            
            # Create a CSV copy for convenience
            csv_path = output_path.with_suffix(".csv")
            try:
                shutil.copy2(output_path, csv_path)
                print(f"CSV Copy        : {csv_path.name} (Successfully created)")
            except Exception as e:
                print(f"CSV Copy        : Failed to create copy ({e})")
        else:
            print("⚠️  Warning: Output file is empty or missing!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

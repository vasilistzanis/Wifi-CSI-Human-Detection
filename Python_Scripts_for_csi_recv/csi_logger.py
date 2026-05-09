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

from csi_parser import configure_console_output
configure_console_output()




# Cross-platform defaults
import config
DEFAULT_PORT = config.SERIAL_PORT


# Mismatch causes garbled output and zero valid frames.
DEFAULT_BAUD = config.BAUD_RATE
DEFAULT_IDLE_SLEEP = config.LOGGER_IDLE_SLEEP
DEFAULT_FLUSH_INTERVAL = config.LOGGER_FLUSH_INTERVAL
DEFAULT_STATUS_INTERVAL = config.LOGGER_STATUS_INTERVAL
DEFAULT_SERIAL_BUFFER_SIZE = config.RX_BUFFER_SIZE
MAX_FILE_SIZE_MB = config.LOGGER_MAX_FILE_SIZE_MB  # Safety limit to prevent filling disk
CSV_HEADER = (
    "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,"
    "channel,local_timestamp,sig_len,rx_state,len,first_word,data\n"
)


BASE_DIR = Path(__file__).resolve().parent




def parse_args():
    defaults = config.get_script_defaults("csi_logger")
    parser = argparse.ArgumentParser(description="High-speed ESP32 CSI logger")
    parser.add_argument("-p", "--port", default=defaults["port"], help="Serial port, e.g. COM6")
    parser.add_argument("-b", "--baud", type=int, default=defaults["baud"], help="Baud rate")
    parser.add_argument("-l", "--label", default=defaults["label"], help="Dataset label. If omitted, you will be prompted.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=defaults["output_dir"],
        help="Output directory. Relative paths are resolved from this script.",
    )
    parser.add_argument(
        "--idle-sleep",
        type=float,
        default=defaults["idle_sleep"],
        help="Sleep time when the serial buffer is empty.",
    )
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=defaults["flush_interval"],
        help="Seconds between file flushes.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=defaults["status_interval"],
        help="Seconds between progress updates.",
    )
    parser.add_argument(
        "--serial-buffer-size",
        type=int,
        default=defaults["serial_buffer_size"],
        help="Windows RX buffer size in bytes.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=defaults["max_size_mb"],
        help="Maximum file size in MB (safety limit).",
    )
    parser.add_argument(
        "-w", "--wait",
        type=int,
        default=defaults["wait"],
        help="Countdown in seconds before recording starts. Set to 0 to disable. (default: 5)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=defaults["duration"],
        help="Auto-stop recording after N seconds. Set to 0 to record continuously until Ctrl+C. (default: 0)",
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
    return input("Enter capture label (e.g. walk_activity_1, fall_3, empty): ")


def export_csv_with_header(source_path: Path, csv_path: Path) -> None:
    """Create a spreadsheet-friendly CSV export with a header row."""
    with open(source_path, "r", encoding="utf-8", errors="ignore") as src, \
            open(csv_path, "w", encoding="utf-8", newline="\n") as dst:
        first_line = src.readline()
        if first_line.startswith("type,"):
            dst.write(first_line if first_line.endswith("\n") else first_line + "\n")
        else:
            dst.write(CSV_HEADER)
            if first_line:
                dst.write(first_line if first_line.endswith("\n") else first_line + "\n")
        shutil.copyfileobj(src, dst)




def main() -> int:
    args = parse_args()


    # -- Port Validation ---------------------------------------------------
    if not validate_port(args.port):
        print(f"[ERROR] Port '{args.port}' not found or not accessible.")
        available = list_available_ports()
        if available:
            print("\n[INFO] Available ports:")
            for port in available:
                print(f"   {port}")
            print("\nUse -p <PORT> to specify a different port.")
        else:
            print("[WARNING]  No serial ports detected. Check your USB connection.")
        return 1


    # -- Label Validation --------------------------------------------------
    raw_label = args.label if args.label else prompt_for_label()
    label = sanitize_label(raw_label)
    if not label:
        print("[ERROR] No valid label was provided. Exiting.")
        return 1


    # -- Output Directory Setup --------------------------------------------
    output_dir = resolve_output_dir(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as exc:
        print(f"[ERROR] Cannot create output directory {output_dir}: {exc}")
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
        # -- Open Serial Port ----------------------------------------------
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


        # -- Countdown Timer -----------------------------------------------
        if args.wait > 0:
            print(f"[WAIT] Starting in {args.wait} seconds... (Get into position!)")
            for i in range(args.wait, 0, -1):
                print(f"   {i}...")
                time.sleep(1)
            print("[START]  GO! (Recording started)\n")
        

        # Flush serial buffer and reset timers — excludes pre-recording movement
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        start_time = time.monotonic()
        last_flush = start_time
        last_status = start_time


        with open(output_path, "wb", buffering=1024*1024) as handle:
            capture_started = True


            while True:
                # Safety net: catches any edge case where bytes_written
                # already equals max_bytes at loop entry
                if bytes_written >= max_bytes:
                    print(f"\n[WARNING]  Reached maximum file size ({args.max_size_mb} MB)")
                    print("   Stopping capture.")
                    break


                waiting = ser.in_waiting
                if waiting <= 0:
                    time.sleep(args.idle_sleep)
                    continue


                chunk = ser.read(waiting)
                if not chunk:
                    continue


                if bytes_written + len(chunk) > max_bytes:
                    print(f"\n[WARNING]  Reached maximum file size ({args.max_size_mb} MB)")
                    print("   Stopping capture to prevent disk overflow.")
                    break


                handle.write(chunk)
                bytes_written += len(chunk)


                now = time.monotonic()


                # -- Periodic File Flush -----------------------------------
                if now - last_flush >= args.flush_interval:
                    handle.flush()
                    last_flush = now


                # -- Auto-Stop Timer ---------------------------------------
                if args.duration > 0 and (now - start_time) >= args.duration:
                    print(f"\n[OK]  Auto-stop reached ({args.duration} seconds).")
                    break


                # -- Status Updates ----------------------------------------
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
        print(f"\n[ERROR] Serial error on {args.port}: {exc}")
        print("   Check that the device is connected and not in use.")
        return 1
    except PermissionError as exc:
        print(f"\n[ERROR] Permission denied: {exc}")
        print(f"   Cannot write to {output_path}")
        return 1
    except KeyboardInterrupt:
        print("\n\n[STOP]  Capture interrupted by user (Ctrl+C)")
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")
        return 1
    finally:
        if ser is not None and ser.is_open:
            ser.close()


    # -- Final Statistics --------------------------------------------------
    if capture_started:
        elapsed = max(time.monotonic() - start_time, 1e-6)
        kb_written = bytes_written / 1024.0
        avg_rate = kb_written / elapsed
        print("\n")
        print("=" * 48)
        print(f"[OK] Capture finished")
        print("=" * 48)
        print(f"Total written   : {kb_written:.1f} KB")
        print(f"Elapsed time    : {elapsed:.2f} s")
        print(f"Average rate    : {avg_rate:.1f} KB/s")
        print(f"File location   : {output_path}")
        

        # Verify file was written
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"File size       : {output_path.stat().st_size / 1024:.1f} KB")
            

            # Create a true CSV export with a header row for spreadsheet tools
            csv_path = output_path.with_suffix(".csv")
            try:
                export_csv_with_header(output_path, csv_path)
                print(f"CSV Export      : {csv_path.name} (Header added successfully)")
            except Exception as e:
                print(f"CSV Export      : Failed to create export ({e})")
        else:
            print("[WARNING]  Warning: Output file is empty or missing!")


    return 0




if __name__ == "__main__":
    sys.exit(main())

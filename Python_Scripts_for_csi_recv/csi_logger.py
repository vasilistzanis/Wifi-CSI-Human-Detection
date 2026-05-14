#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
High-speed ESP32 CSI logger with improved validation and error handling.

Activity mode (--mode):
  Preset for a specific activity: fall, sit, walk, idle.
  Automatically sets output directory, duration, and auto-numbers labels.
  Use -n N to record N sessions in sequence without restarting the script.

  Examples:
    python csi_logger.py --mode fall -n 20
    python csi_logger.py --mode sit  -n 20
    python csi_logger.py --mode walk -n 5
    python csi_logger.py --mode idle -n 3

Impact marker:
  During any recording, press SPACE at the moment of the action.
  Timestamps are saved to a .meta.json sidecar file next to the .txt recording.
"""


import argparse
import json
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




import config

CSV_HEADER = (
    "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,"
    "channel,local_timestamp,sig_len,rx_state,len,first_word,data\n"
)

BASE_DIR = Path(__file__).resolve().parent
_MODE_CHOICES = list(config.LOGGER_ACTIVITY_PRESETS.keys())




def parse_args():
    defaults = config.get_script_defaults("csi_logger")
    parser = argparse.ArgumentParser(
        description="High-speed ESP32 CSI logger",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-p", "--port", default=defaults["port"], help="Serial port, e.g. COM6")
    parser.add_argument("-b", "--baud", type=int, default=defaults["baud"], help="Baud rate")
    parser.add_argument("-l", "--label", default=defaults["label"],
                        help="Dataset label. If omitted, you will be prompted.")
    parser.add_argument("-o", "--output-dir", default=defaults["output_dir"],
                        help="Output directory. Relative paths are resolved from this script.")
    parser.add_argument("--idle-sleep", type=float, default=defaults["idle_sleep"],
                        help="Sleep time when the serial buffer is empty.")
    parser.add_argument("--flush-interval", type=float, default=defaults["flush_interval"],
                        help="Seconds between file flushes.")
    parser.add_argument("--status-interval", type=float, default=defaults["status_interval"],
                        help="Seconds between progress updates.")
    parser.add_argument("--serial-buffer-size", type=int, default=defaults["serial_buffer_size"],
                        help="Windows RX buffer size in bytes.")
    parser.add_argument("--max-size-mb", type=int, default=defaults["max_size_mb"],
                        help="Maximum file size in MB (safety limit).")
    parser.add_argument("-w", "--wait", type=int, default=defaults["wait"],
                        help="Countdown in seconds before recording starts. 0 to disable. (default: 5)")
    parser.add_argument("-d", "--duration", type=int, default=defaults["duration"],
                        help="Auto-stop after N seconds. 0 = continuous until Ctrl+C. (default: 0)")

    preset_lines = "\n".join(
        f"  {k:6s}: dir={v['output_dir']}, duration={v['duration']}s, label={v['label_prefix']}_NNN"
        for k, v in config.LOGGER_ACTIVITY_PRESETS.items()
    )
    parser.add_argument(
        "--mode",
        choices=_MODE_CHOICES,
        default=defaults["mode"],
        metavar="MODE",
        help=(
            f"Activity preset (choices: {', '.join(_MODE_CHOICES)}).\n"
            "Sets output dir, duration, and auto-numbers labels.\n"
            f"{preset_lines}"
        ),
    )
    parser.add_argument("-n", "--repeat", type=int, default=defaults["repeat"],
                        help="Number of recordings in sequence (default: 1).")
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


def list_available_ports() -> list[str]:
    return [p.device for p in list_ports.comports()]


def validate_port(port: str) -> bool:
    return port in list_available_ports()


def safe_set_buffer_size(ser: serial.Serial, rx_size: int) -> None:
    if os.name != "nt" or not hasattr(ser, "set_buffer_size"):
        return
    try:
        ser.set_buffer_size(rx_size=rx_size)
    except Exception:
        pass


def prompt_for_label() -> str:
    return input("Enter capture label (e.g. walk_activity_1, no_activity_2): ")


def _drain_kbd() -> None:
    """Discard any keys queued before recording starts (Windows only)."""
    if os.name == "nt":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()


def _poll_kbd() -> bytes | None:
    """Non-blocking single-key read. Returns the raw byte or None."""
    if os.name == "nt":
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getch()
    return None


def _next_auto_label(output_dir: Path, prefix: str) -> str:
    """Auto-number next recording: prefix_001, prefix_002, ..."""
    max_num = 0
    for f in output_dir.glob(f"{prefix}_*.txt"):
        m = re.match(rf"{re.escape(prefix)}_(\d{{1,4}})(?:_|$)", f.stem)
        if m:
            n = int(m.group(1))
            if n < 1000:  # unix timestamps are 10 digits, our numbers are ≤999
                max_num = max(max_num, n)
    return f"{prefix}_{max_num + 1:03d}"


def export_csv_with_header(source_path: Path, csv_path: Path) -> None:
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




def _do_capture(ser, args, label, output_dir, max_bytes, idx, total):
    """Run one recording session. Returns (output_path, bytes_written, elapsed, event_markers)."""
    output_path = output_dir / f"{label}_{int(time.time())}.txt"
    bytes_written = 0
    event_markers: list[float] = []
    start_time = time.monotonic()
    last_flush = start_time
    last_status = start_time

    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    if total > 1:
        print(f"\n{'─' * 48}")
        print(f"  Recording {idx}/{total}  —  {label}")
        print(f"{'─' * 48}")
    else:
        print(f"Output : {output_path}")
        if args.duration > 0:
            print(f"Duration: {args.duration} s")

    if args.wait > 0:
        print(f"\n[WAIT] Starting in {args.wait} seconds... (Get into position!)")
        for i in range(args.wait, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        print("[START]  GO! (Recording started)")

    print("  >> Press SPACE at the moment of the action to mark it <<\n")

    # Flush stale serial data and keyboard buffer accumulated during countdown
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    _drain_kbd()

    start_time = time.monotonic()
    last_flush = start_time
    last_status = start_time

    with open(output_path, "wb", buffering=1024 * 1024) as handle:
        while True:
            if bytes_written >= max_bytes:
                print(f"\n[WARNING]  Reached maximum file size ({args.max_size_mb} MB)")
                print("   Stopping capture.")
                break

            waiting = ser.in_waiting
            if waiting <= 0:
                time.sleep(args.idle_sleep)
            else:
                chunk = ser.read(waiting)
                if chunk:
                    if bytes_written + len(chunk) > max_bytes:
                        print(f"\n[WARNING]  Reached maximum file size ({args.max_size_mb} MB)")
                        print("   Stopping capture to prevent disk overflow.")
                        break
                    handle.write(chunk)
                    bytes_written += len(chunk)

            now = time.monotonic()

            # Event marker: Space or Enter
            key = _poll_kbd()
            if key in (b" ", b"\r", b"\n"):
                t = now - start_time
                event_markers.append(round(t, 3))
                print(f"\n  [MARK #{len(event_markers)}] Event at t = {t:.2f}s", flush=True)

            if now - last_flush >= args.flush_interval:
                handle.flush()
                last_flush = now

            if args.duration > 0 and (now - start_time) >= args.duration:
                print(f"\n[OK]  Auto-stop reached ({args.duration} seconds).")
                break

            if now - last_status >= args.status_interval:
                elapsed = max(now - start_time, 1e-6)
                kb_written = bytes_written / 1024.0
                avg_rate = kb_written / elapsed
                percent_full = (bytes_written / max_bytes) * 100
                remaining = (
                    f" | {args.duration - elapsed:.0f}s left"
                    if args.duration > 0 else ""
                )
                marks = f" | marks: {len(event_markers)}" if event_markers else ""
                print(
                    f"\rCaptured: {kb_written:10.1f} KB "
                    f"({percent_full:5.1f}%) | "
                    f"Avg rate: {avg_rate:8.1f} KB/s{remaining}{marks}",
                    end="",
                    flush=True,
                )
                last_status = now

    elapsed = max(time.monotonic() - start_time, 1e-6)
    return output_path, bytes_written, elapsed, event_markers




def main() -> int:
    args = parse_args()

    # -- Activity mode presets ---------------------------------------------
    preset = config.LOGGER_ACTIVITY_PRESETS.get(args.mode) if args.mode else None
    if preset:
        if args.output_dir == config.LOGGER_OUTPUT_DIR:
            args.output_dir = preset["output_dir"]
        if args.duration == config.LOGGER_DURATION_SECONDS:
            args.duration = preset["duration"]

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

    # -- Output Directory Setup --------------------------------------------
    output_dir = resolve_output_dir(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as exc:
        print(f"[ERROR] Cannot create output directory {output_dir}: {exc}")
        return 1

    max_bytes = args.max_size_mb * 1024 * 1024

    # -- Label Setup -------------------------------------------------------
    if preset and not args.label:
        base_label = None   # auto-generated per recording from preset label_prefix
    elif args.label:
        base_label = sanitize_label(args.label)
        if not base_label:
            print("[ERROR] No valid label was provided. Exiting.")
            return 1
    else:
        raw_label = prompt_for_label()
        base_label = sanitize_label(raw_label)
        if not base_label:
            print("[ERROR] No valid label was provided. Exiting.")
            return 1

    ser = None

    try:
        # -- Open Serial Port ----------------------------------------------
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
        safe_set_buffer_size(ser, args.serial_buffer_size)
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        # -- Session Header ------------------------------------------------
        print("\n" + "=" * 48)
        if args.mode:
            print(f"ESP32-C6 RADAR  —  {args.mode.upper()} RECORDING MODE")
        else:
            print("ESP32-C6 RADAR  —  HIGH SPEED LOGGER")
        print("=" * 48)
        print(f"Port    : {args.port}")
        print(f"Baud    : {args.baud}")
        print(f"Output  : {output_dir}")
        print(f"Max     : {args.max_size_mb} MB")
        if args.duration > 0:
            print(f"Duration: {args.duration} s per recording")
        if args.repeat > 1:
            print(f"Repeat  : {args.repeat} recordings")
        print("Press Ctrl+C to stop.\n")

        # -- Recording Loop ------------------------------------------------
        completed = 0
        for i in range(args.repeat):

            # Generate label for this iteration
            if base_label is None:
                label = _next_auto_label(output_dir, preset["label_prefix"])
            elif args.repeat > 1:
                label = f"{base_label}_{i + 1:03d}"
            else:
                label = base_label

            # Between recordings: wait for user to get back into position
            if i > 0:
                try:
                    input(
                        f"\n[NEXT] Get ready. "
                        f"Press Enter to start recording {i + 1}/{args.repeat} ({label}),"
                        f" or Ctrl+C to stop..."
                    )
                except KeyboardInterrupt:
                    print("\n[STOP]  Stopped early by user.")
                    break

            output_path, bytes_written, elapsed, event_markers = _do_capture(
                ser, args, label, output_dir, max_bytes, i + 1, args.repeat
            )
            completed += 1

            # -- Per-recording Summary -------------------------------------
            if bytes_written > 0:
                kb_written = bytes_written / 1024.0
                avg_rate = kb_written / elapsed
                print(f"\n{'=' * 48}")
                if args.repeat > 1:
                    remaining = args.repeat - (i + 1)
                    print(f"[OK] Recording {i + 1}/{args.repeat} done  ({remaining} remaining)")
                else:
                    print(f"[OK] Capture finished")
                print(f"{'=' * 48}")
                print(f"Total written   : {kb_written:.1f} KB")
                print(f"Elapsed time    : {elapsed:.2f} s")
                print(f"Average rate    : {avg_rate:.1f} KB/s")
                print(f"File location   : {output_path}")

                if event_markers:
                    print(f"Event markers   : {[f'{t:.2f}s' for t in event_markers]}")

                if output_path.exists() and output_path.stat().st_size > 0:
                    print(f"File size       : {output_path.stat().st_size / 1024:.1f} KB")

                    # Meta sidecar: write when in activity mode or if markers were set
                    if args.mode or event_markers:
                        meta_path = output_path.with_suffix(".meta.json")
                        meta = {
                            "label": label,
                            "mode": args.mode,
                            "duration_s": round(elapsed, 3),
                            "event_markers_s": event_markers,
                            "event_count": len(event_markers),
                            "recording_unix": int(output_path.stem.split("_")[-1])
                                              if output_path.stem.split("_")[-1].isdigit()
                                              else int(time.time()),
                        }
                        try:
                            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                            print(f"Meta sidecar    : {meta_path.name}")
                        except Exception as e:
                            print(f"Meta sidecar    : Failed ({e})")

                    csv_path = output_path.with_suffix(".csv")
                    try:
                        export_csv_with_header(output_path, csv_path)
                        print(f"CSV Export      : {csv_path.name} (Header added successfully)")
                    except Exception as e:
                        print(f"CSV Export      : Failed to create export ({e})")
                else:
                    print("[WARNING]  Output file is empty or missing!")
            else:
                print("[WARNING]  Nothing was captured for this recording.")

        # -- Session Summary -----------------------------------------------
        if args.repeat > 1:
            print(f"\n{'=' * 48}")
            print(f"SESSION DONE: {completed}/{args.repeat} recordings saved to {output_dir}")
            print(f"{'=' * 48}")

    except serial.SerialException as exc:
        print(f"\n[ERROR] Serial error on {args.port}: {exc}")
        print("   Check that the device is connected and not in use.")
        return 1
    except PermissionError as exc:
        print(f"\n[ERROR] Permission denied: {exc}")
        print(f"   Cannot write to {output_dir}")
        return 1
    except KeyboardInterrupt:
        print("\n\n[STOP]  Capture interrupted by user (Ctrl+C)")
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")
        return 1
    finally:
        if ser is not None and ser.is_open:
            ser.close()

    return 0




if __name__ == "__main__":
    sys.exit(main())

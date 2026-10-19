from __future__ import annotations

import argparse
import os
import queue
import threading
import time

import serial
from serial.tools import list_ports

import config
from csi_forwarding import send_with_retry, validate_server_url
from csi_serial import iter_csi_lines


def _read_serial(
    serial_port: serial.Serial,
    frames: queue.Queue[str],
    stopped: threading.Event,
    errors: queue.Queue[Exception],
) -> None:
    try:
        for line in iter_csi_lines(serial_port, stopped):
            try:
                frames.put(line, timeout=0.5)
            except queue.Full:
                errors.put(RuntimeError(
                    "CSI frame queue is full; stopping rather than silently losing data."
                ))
                stopped.set()
                return
    except serial.SerialException as exc:
        errors.put(exc)
        stopped.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward CSI serial frames from the ESP32 to the cloud API."
    )
    parser.add_argument("--port", default=config.SERIAL_PORT, help="ESP32 serial port.")
    parser.add_argument("--baud", type=int, default=config.BAUD_RATE, help="Serial baud rate.")
    parser.add_argument(
        "--server",
        default=os.environ.get("CSI_SERVER_URL"),
        help="Cloud API base URL, e.g. https://csi.example.com.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Frames per HTTPS request (1-100)."
    )
    parser.add_argument(
        "--list-ports", action="store_true", help="List available serial devices and exit."
    )
    args = parser.parse_args()
    if args.list_ports:
        return args
    if not args.server and not args.list_ports:
        parser.error("Set CSI_SERVER_URL or pass --server.")
    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100.")
    try:
        validate_server_url(args.server)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = parse_args()
    if args.list_ports:
        ports = list(list_ports.comports())
        if not ports:
            print("No serial ports found. Connect the ESP32 with a USB data cable.")
            return 0
        for port in ports:
            print(f"{port.device}: {port.description}")
        return 0

    token = os.environ.get("CSI_EDGE_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Set CSI_EDGE_TOKEN to the same secret used by the API (32+ chars).")

    try:
        serial_port = serial.Serial(args.port, args.baud, timeout=0.2)
    except serial.SerialException as exc:
        raise SystemExit(
            f"Cannot open serial port {args.port}: {exc}. "
            "Use --list-ports to find the Linux device (usually /dev/ttyACM0 "
            "or /dev/ttyUSB0). On Linux, check membership in the device's "
            "group, commonly 'dialout', then log out and back in."
        ) from None
    frames: queue.Queue[str] = queue.Queue(maxsize=5000)
    errors: queue.Queue[Exception] = queue.Queue()
    stopped = threading.Event()
    reader = threading.Thread(
        target=_read_serial,
        args=(serial_port, frames, stopped, errors),
        name="csi-serial-reader",
        daemon=True,
    )
    reader.start()
    print(f"Forwarding CSI from {args.port} to {args.server.rstrip('/')}.")

    try:
        while not stopped.is_set() or not frames.empty():
            try:
                first = frames.get(timeout=0.25)
            except queue.Empty:
                if not errors.empty():
                    raise errors.get()
                continue

            batch = [first]
            deadline = time.monotonic() + 0.1
            while len(batch) < args.batch_size and time.monotonic() < deadline:
                try:
                    batch.append(frames.get(timeout=max(0.01, deadline - time.monotonic())))
                except queue.Empty:
                    break
            accepted = send_with_retry(args.server, token, batch)
            if accepted != len(batch):
                print(f"Cloud accepted {accepted} of {len(batch)} frames.")
            else:
                print(f"Sent {accepted} CSI frames.")

        if not errors.empty():
            raise errors.get()
    except KeyboardInterrupt:
        print("Stopping collector.")
    finally:
        stopped.set()
        serial_port.close()
        reader.join(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

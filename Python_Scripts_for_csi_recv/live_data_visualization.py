#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import threading
from dataclasses import dataclass, field


import numpy as np
import pyqtgraph as pg
import serial
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from pyqtgraph import PlotWidget, ScatterPlotItem


def configure_console_output() -> None:
    """Avoid UnicodeEncodeError on legacy Windows console encodings."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


configure_console_output()

DEFAULT_BAUD = 2_000_000
DEFAULT_BUFFER_SIZE = 200
DEFAULT_SUBCARRIERS = 128
DEFAULT_REFRESH_MS = 50
DEFAULT_SERIAL_TIMEOUT = 0.25
DEFAULT_SERIAL_BUFFER_SIZE = 2_000_000
RECV_FIELD_COUNT = 15



def split_recv_fields(line: str):
    if not line.startswith("CSI_DATA"):
        return None

    parts = [part.strip() for part in line.strip().split(",", RECV_FIELD_COUNT - 1)]
    if len(parts) != RECV_FIELD_COUNT:
        return None

    for idx in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
        try:
            int(parts[idx])
        except ValueError:
            return None

    return parts


def parse_args():
    parser = argparse.ArgumentParser(description="Live CSI viewer for ESP32 CSI frames")
    parser.add_argument("-p", "--port", required=True, help="Serial port, e.g. COM6")
    parser.add_argument("-b", "--baud", type=int, default=DEFAULT_BAUD, help="Baud rate")
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_SIZE,
        help="Number of frames kept in the circular buffer.",
    )
    parser.add_argument(
        "--subcarriers",
        type=int,
        default=DEFAULT_SUBCARRIERS,
        help="Expected number of complex subcarriers per CSI frame.",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=DEFAULT_REFRESH_MS,
        help="UI refresh interval in milliseconds.",
    )
    parser.add_argument(
        "--serial-timeout",
        type=float,
        default=DEFAULT_SERIAL_TIMEOUT,
        help="Serial readline timeout in seconds.",
    )
    parser.add_argument(
        "--serial-buffer-size",
        type=int,
        default=DEFAULT_SERIAL_BUFFER_SIZE,
        help="Windows RX buffer size in bytes.",
    )
    return parser.parse_args()


def safe_set_buffer_size(ser: serial.Serial, rx_size: int) -> None:
    if os.name != "nt" or not hasattr(ser, "set_buffer_size"):
        return

    try:
        ser.set_buffer_size(rx_size=rx_size)
    except Exception:
        pass


def extract_payload(line: str):
    parts = split_recv_fields(line)
    if parts is None:
        return None

    payload = parts[14].strip().strip('"')
    if not payload.startswith("[") or not payload.endswith("]"):
        return None
    return payload[1:-1].strip()


def extract_seq(line: str):
    parts = split_recv_fields(line)
    return int(parts[1]) if parts is not None else None


def parse_csi_frame(line: str, subcarriers: int):
    payload = extract_payload(line)
    if not payload:
        return None

    expected_values = subcarriers * 2
    token_count = payload.count(",") + 1
    values = np.fromstring(payload, sep=",", dtype=np.float32)

    if token_count != expected_values or values.size != expected_values:
        return None

    imag = values[0::2]
    real = values[1::2]
    return (real + 1j * imag).astype(np.complex64)


@dataclass
class CSIState:
    buffer_size: int
    subcarriers: int
    buffer: np.ndarray = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    write_idx: int = 0
    frame_count: int = 0
    dropped_count: int = 0
    connected: bool = False
    last_error: str = ""
    last_seq: int | None = None
    seq_sample_count: int = 0
    missing_seq_count: int = 0
    gap_event_count: int = 0
    non_monotonic_count: int = 0
    last_gap_size: int = 0

    def __post_init__(self):
        self.buffer = np.zeros((self.buffer_size, self.subcarriers), dtype=np.complex64)

    def push_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            self.buffer[self.write_idx % self.buffer_size] = frame
            self.write_idx += 1
            self.frame_count += 1

    def update_seq(self, seq: int) -> None:
        with self.lock:
            if self.last_seq is not None:
                if seq > self.last_seq + 1:
                    gap_size = seq - self.last_seq - 1
                    self.missing_seq_count += gap_size
                    self.gap_event_count += 1
                    self.last_gap_size = gap_size
                elif seq <= self.last_seq:
                    self.non_monotonic_count += 1

            self.last_seq = seq
            self.seq_sample_count += 1

    def mark_drop(self) -> None:
        with self.lock:
            self.dropped_count += 1

    def set_connected(self, connected: bool) -> None:
        with self.lock:
            self.connected = connected

    def set_error(self, message: str) -> None:
        with self.lock:
            self.last_error = message

    def snapshot(self):
        with self.lock:
            latest_frame = None
            if self.frame_count > 0:
                latest_idx = (self.write_idx - 1) % self.buffer_size
                latest_frame = self.buffer[latest_idx].copy()

            return {
                "latest_frame": latest_frame,
                "frame_count": self.frame_count,
                "dropped_count": self.dropped_count,
                "connected": self.connected,
                "last_error": self.last_error,
                "buffer_fill": min(self.frame_count, self.buffer_size),
                "buffer_size": self.buffer_size,
                "last_seq": self.last_seq,
                "missing_seq_count": self.missing_seq_count,
                "gap_event_count": self.gap_event_count,
                "non_monotonic_count": self.non_monotonic_count,
                "last_gap_size": self.last_gap_size,
                "loss_percent": (
                    (self.missing_seq_count / (self.seq_sample_count + self.missing_seq_count)) * 100.0
                    if (self.seq_sample_count + self.missing_seq_count) > 0
                    else 0.0
                ),
            }


class SerialReader(QThread):
    def __init__(
        self,
        port: str,
        baud: int,
        timeout: float,
        rx_buffer_size: int,
        state: CSIState,
        stop_event: threading.Event,
    ):
        super().__init__()
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.rx_buffer_size = rx_buffer_size
        self.state = state
        self.stop_event = stop_event

    def run(self):
        ser = None

        try:
            ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            safe_set_buffer_size(ser, self.rx_buffer_size)

            try:
                ser.reset_input_buffer()
            except Exception:
                pass

            self.state.set_connected(True)
            self.state.set_error("")
            print(f"Connected to {self.port} at {self.baud} baud.")

            while not self.stop_event.is_set():
                try:
                    raw = ser.readline()
                except serial.SerialException as exc:
                    message = f"Serial read error on {self.port}: {exc}"
                    self.state.set_error(message)
                    print(message)
                    break

                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("CSI_DATA"):
                    continue

                seq = extract_seq(line)
                if seq is not None:
                    self.state.update_seq(seq)

                frame = parse_csi_frame(line, self.state.subcarriers)
                if frame is None:
                    self.state.mark_drop()
                    continue

                self.state.push_frame(frame)

        except serial.SerialException as exc:
            message = f"Could not open {self.port}: {exc}"
            self.state.set_error(message)
            print(message)
        finally:
            self.state.set_connected(False)

            if ser is not None and ser.is_open:
                ser.close()

            print("Serial port closed.")


class CSIWindow(QWidget):
    def __init__(self, state: CSIState, port: str, refresh_ms: int):
        super().__init__()
        self.state = state
        self.port = port

        self.resize(1100, 900)
        self.setWindowTitle(f"ESP32-C6 CSI Live Viewer ({port})")
        self.setStyleSheet("background-color: #0f0f1e; color: #e8e8e8;")

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.status_label = QLabel(f"Waiting for CSI data on {port}...")
        self.status_label.setStyleSheet(
            "color: #00ff88; font-family: Consolas, monospace; font-size: 13px; padding: 5px;"
        )
        layout.addWidget(self.status_label)

        self.plot_amp = PlotWidget(self)
        self.plot_amp.setTitle("Live CSI Amplitude", color="#f7b731", size="12pt")
        self.plot_amp.setLabel("left", "Amplitude")
        self.plot_amp.setLabel("bottom", "Subcarrier Index")
        self.plot_amp.showGrid(x=True, y=True, alpha=0.3)
        self.curve_amp = self.plot_amp.plot(pen=pg.mkPen("#f7b731", width=1.5))
        layout.addWidget(self.plot_amp)

        self.plot_phase = PlotWidget(self)
        self.plot_phase.setTitle("Live CSI Phase", color="#26de81", size="12pt")
        self.plot_phase.setLabel("left", "Phase (rad)")
        self.plot_phase.setLabel("bottom", "Subcarrier Index")
        self.plot_phase.setYRange(-np.pi, np.pi)
        self.plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self.curve_phase = self.plot_phase.plot(pen=pg.mkPen("#26de81", width=1.5))
        layout.addWidget(self.plot_phase)

        self.plot_iq = PlotWidget(self)
        self.plot_iq.setTitle("IQ Constellation", color="#fc5c65", size="12pt")
        self.plot_iq.setLabel("left", "Q")
        self.plot_iq.setLabel("bottom", "I")
        self.plot_iq.getViewBox().setAspectLocked(True)
        self.plot_iq.showGrid(x=True, y=True, alpha=0.3)
        self.scatter = ScatterPlotItem(
            size=4,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(252, 92, 101, 180),
        )
        self.plot_iq.addItem(self.scatter)
        layout.addWidget(self.plot_iq)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(refresh_ms)

    def update_plots(self):
        snapshot = self.state.snapshot()

        status = (
            f"Port: {self.port} "
            f"({'connected' if snapshot['connected'] else 'disconnected'}) | "
            f"Seq: {snapshot['last_seq'] if snapshot['last_seq'] is not None else '-'} | "
            f"Missing: {snapshot['missing_seq_count']} ({snapshot['loss_percent']:.2f}%) | "
            f"Frames: {snapshot['frame_count']} | "
            f"Dropped: {snapshot['dropped_count']} | "
            f"Buffer: {snapshot['buffer_fill']}/{snapshot['buffer_size']}"
        )

        if snapshot["gap_event_count"] > 0:
            status += f" | Gaps: {snapshot['gap_event_count']} (last: {snapshot['last_gap_size']})"

        if snapshot["non_monotonic_count"] > 0:
            status += f" | Seq resets: {snapshot['non_monotonic_count']}"

        if snapshot["last_error"]:
            status += f" | {snapshot['last_error']}"

        self.status_label.setText(status)

        latest_frame = snapshot["latest_frame"]
        if latest_frame is None:
            return

        active_indices = np.flatnonzero(np.abs(latest_frame) > 0)
        if active_indices.size == 0:
            return

        data = latest_frame[active_indices]
        amplitude = np.abs(data)
        phase = np.angle(data)

        self.curve_amp.setData(active_indices, amplitude)
        self.curve_phase.setData(active_indices, phase)
        self.scatter.setData(x=np.real(data), y=np.imag(data))


def main():
    args = parse_args()

    pg.setConfigOptions(antialias=True)

    state = CSIState(
        buffer_size=args.buffer_size,
        subcarriers=args.subcarriers,
    )

    app = QApplication(sys.argv)
    stop_event = threading.Event()

    reader = SerialReader(
        port=args.port,
        baud=args.baud,
        timeout=args.serial_timeout,
        rx_buffer_size=args.serial_buffer_size,
        state=state,
        stop_event=stop_event,
    )

    window = CSIWindow(state=state, port=args.port, refresh_ms=args.refresh_ms)

    app.aboutToQuit.connect(stop_event.set)

    reader.start()
    window.show()

    if hasattr(app, "exec"):
        exit_code = app.exec()
    else:
        exit_code = app.exec_()

    stop_event.set()
    reader.wait(2000)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

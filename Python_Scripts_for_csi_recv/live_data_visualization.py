#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESP32-C6 CSI Ultra-Fast Live Viewer (Thesis Grade) — Final Version
------------------------------------------------------------------
Features: 
- Thread-safe circular buffer (O(1) complexity)
- Race condition protection (Locking mechanism)
- Live Amplitude, Phase & IQ Constellation plots
- Optimized string parsing (No JSON overhead)
- Automated cleanup & Graceful shutdown
"""

import sys
import argparse
import numpy as np
import serial
import threading
from PyQt5.Qt import *
from PyQt5 import QtCore
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ScatterPlotItem
from PyQt5.QtCore import QThread

# ==============================================================================
# GLOBAL DATA & CONFIGURATION
# ==============================================================================
CSI_DATA_INDEX   = 200   # Πλήθος frames που κρατάμε στη μνήμη για το γράφημα
CSI_DATA_COLUMNS = 128   # 128 complex subcarriers (256 τιμές I/Q)

# Buffers και Threading Safety
csi_data_complex = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
data_lock        = threading.Lock()
write_idx        = 0
frame_count      = 0
dropped_count    = 0

# ==============================================================================
# GUI WINDOW CLASS
# ==============================================================================
class CSIWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1100, 900)
        self.setWindowTitle("ESP32-C6 CSI Live Viewer — Thesis Grade")
        self.setStyleSheet("background-color: #0f0f1e;")

        # Layout setup
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Status Label (Stats)
        self.status_label = QLabel("⏳ Αναμονή δεδομένων από τη σειριακή θύρα...")
        self.status_label.setStyleSheet("color: #00ff88; font-family: Monospace; font-size: 13px; padding: 5px;")
        main_layout.addWidget(self.status_label)

        # ---- Amplitude Plot ----
        self.plot_amp = PlotWidget(self)
        self.plot_amp.setTitle("Live CSI Amplitude", color='#f7b731', size='12pt')
        self.plot_amp.showGrid(x=True, y=True, alpha=0.3)
        self.curve_amp = self.plot_amp.plot(pen=pg.mkPen('#f7b731', width=1.5))
        main_layout.addWidget(self.plot_amp)

        # ---- Phase Plot ----
        self.plot_phase = PlotWidget(self)
        self.plot_phase.setTitle("Live CSI Phase (Radians)", color='#26de81', size='12pt')
        self.plot_phase.setYRange(-np.pi, np.pi)
        self.plot_phase.showGrid(x=True, y=True, alpha=0.3)
        self.curve_phase = self.plot_phase.plot(pen=pg.mkPen('#26de81', width=1.5))
        main_layout.addWidget(self.plot_phase)

        # ---- IQ Plot ----
        self.plot_iq = PlotWidget(self)
        self.plot_iq.setTitle("IQ Constellation Map", color='#fc5c65', size='12pt')
        self.plot_iq.getViewBox().setAspectLocked(True)
        self.plot_iq.showGrid(x=True, y=True, alpha=0.3)
        self.scatter = ScatterPlotItem(size=4, pen=pg.mkPen(None), brush=pg.mkBrush(252, 92, 101, 180))
        self.plot_iq.addItem(self.scatter)
        main_layout.addWidget(self.plot_iq)

        # Refresh Timer (~20 FPS για ομαλή κίνηση)
        self._last_frame_processed = -1
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50) 

    def update_plots(self):
        global csi_data_complex, write_idx, frame_count, dropped_count

        # Thread-safe ανάγνωση του τελευταίου πακέτου
        with data_lock:
            current_write_pos = write_idx
            current_frame_total = frame_count
            latest_idx = (current_write_pos - 1) % CSI_DATA_INDEX
            data = csi_data_complex[latest_idx, :].copy()

        # Update Status Bar
        if current_frame_total != self._last_frame_processed:
            self._last_frame_processed = current_frame_total
            self.status_label.setText(
                f"📊 Frames: {current_frame_total} | ⚠️ Dropped: {dropped_count} | "
                f"Buffer: {min(current_frame_total, CSI_DATA_INDEX)}/{CSI_DATA_INDEX}"
            )

        # Φιλτράρισμα Null Subcarriers (εκεί που το Amplitude είναι 0)
        active_mask = np.abs(data) > 0
        data_clean = data[active_mask]

        if len(data_clean) == 0:
            return

        # Ενημέρωση Γραφημάτων
        self.curve_amp.setData(np.abs(data_clean))
        self.curve_phase.setData(np.angle(data_clean))
        self.scatter.setData(x=np.real(data_clean), y=np.imag(data_clean))

# ==============================================================================
# SERIAL PROCESSING THREAD
# ==============================================================================
def serial_reader_task(port, stop_event):
    global csi_data_complex, write_idx, frame_count, dropped_count

    try:
        ser = serial.Serial(port=port, baudrate=2000000, timeout=1)
    except Exception as e:
        print(f"❌ Error opening port: {e}")
        return

    if ser.is_open:
        print(f"✅ Connected to {port} at 2Mbps. Radar is LIVE.")

    while not stop_event.is_set():
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line.startswith("CSI_DATA"):
                continue

            # Parsing των δεδομένων [...]
            start = line.find('['); end = line.find(']')
            if start == -1 or end == -1: continue
            
            raw_vals = line[start+1:end].split(',')
            num_list = [int(x) for x in raw_vals]

            # Κατασκευή του μιγαδικού frame
            current_frame = np.zeros(CSI_DATA_COLUMNS, dtype=np.complex64)
            idx = 0
            for i in range(0, len(num_list), 2):
                if idx >= CSI_DATA_COLUMNS: break
                # ESP32-C6 Format: [Imag, Real, Imag, Real...]
                imag = num_list[i]
                real = num_list[i+1]
                current_frame[idx] = complex(real, imag)
                idx += 1

            # Thread-safe εγγραφή στο Circular Buffer
            with data_lock:
                csi_data_complex[write_idx % CSI_DATA_INDEX] = current_frame
                write_idx += 1
                frame_count += 1

        except Exception:
            dropped_count += 1
            continue

    ser.close()
    print("🔌 Serial port closed gracefully.")

class SerialThread(QThread):
    def __init__(self, port, stop_event):
        super().__init__()
        self.port = port
        self.stop_event = stop_event
    def run(self):
        serial_reader_task(self.port, self.stop_event)

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', required=True, help="π.χ. COM6")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    stop_signal = threading.Event()

    # Start Background Thread
    backend = SerialThread(args.port, stop_signal)
    backend.start()

    # Start UI
    window = CSIWindow()
    window.show()

    # Loop until window close
    exit_code = app.exec()
    
    # Cleanup
    stop_signal.set()
    backend.wait(1000)
    sys.exit(exit_code)
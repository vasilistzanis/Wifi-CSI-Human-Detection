#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import json
import argparse
import numpy as np
import serial
from io import StringIO

from PyQt5.Qt import *
from PyQt5 import QtCore
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ScatterPlotItem
from PyQt5.QtCore import pyqtSignal, QThread

# ==============================
# GLOBAL DATA BUFFER
# ==============================
CSI_DATA_INDEX = 200
CSI_DATA_COLUMNS = 512  # max safe size

csi_data_complex = np.zeros(
    [CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64
)

# ==============================
# GUI WINDOW
# ==============================
class CSIWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1000, 850) # Αυξήσαμε το ύψος για το 3ο γράφημα
        self.setWindowTitle("CSI Live Viewer (Amplitude, Phase & IQ)")

        # Layout setup
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ---- Amplitude Plot ----
        self.plot_amp = PlotWidget(self)
        self.plot_amp.setTitle("Live CSI Amplitude")
        self.plot_amp.setLabel('left', 'Amplitude')
        self.plot_amp.setLabel('bottom', 'Subcarrier Index')
        self.curve_amp = self.plot_amp.plot(pen='y')
        layout.addWidget(self.plot_amp)

        # ---- Phase Plot (ΝΕΟ) ----
        self.plot_phase = PlotWidget(self)
        self.plot_phase.setTitle("Live CSI Phase (Radians)")
        self.plot_phase.setLabel('left', 'Phase (rad)')
        self.plot_phase.setLabel('bottom', 'Subcarrier Index')
        self.curve_phase = self.plot_phase.plot(pen='g') # Πράσινο χρώμα για τη φάση
        layout.addWidget(self.plot_phase)

        # ---- IQ Plot ----
        self.plot_iq = PlotWidget(self)
        self.plot_iq.setTitle("IQ Constellation")
        self.plot_iq.setLabel('left', 'Q (Imaginary)')
        self.plot_iq.setLabel('bottom', 'I (Real)')
        self.plot_iq.getViewBox().setAspectLocked(True)
        self.scatter = ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 150))
        self.plot_iq.addItem(self.scatter)
        layout.addWidget(self.plot_iq)

        # Timer (γρήγορο refresh)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(50)  # ~20 FPS

    def update_data(self):
        global csi_data_complex

        # Παίρνουμε το τελευταίο frame δεδομένων
        data = csi_data_complex[-1, :]

        # Αφαιρούμε τα μηδενικά (υποφέροντα που δεν έχουν δεδομένα)
        data = data[np.abs(data) > 0]

        if len(data) == 0:
            return

        # Downsample (προαιρετικό, για καλύτερη απόδοση αν είναι πολλά τα δεδομένα)
        # data = data[::2] 

        # ---- 1. Υπολογισμός Amplitude ----
        # $A = \sqrt{I^2 + Q^2}$
        amp = np.abs(data)
        self.curve_amp.setData(amp)

        # ---- 2. Υπολογισμός Phase (ΝΕΟ) ----
        # $\phi = \operatorname{atan2}(Q, I)$
        phase = np.angle(data)
        self.curve_phase.setData(phase)

        # ---- 3. Απεικόνιση IQ ----
        i = np.real(data)
        q = np.imag(data)
        self.scatter.setData(x=i, y=q)


# ==============================
# SERIAL READER
# ==============================
def csi_data_read_parse(port, csv_writer, log_file_fd):
    # Χρησιμοποιούμε το baudrate που ζήτησες
    ser = serial.Serial(port=port, baudrate=921600, timeout=1)

    if ser.isOpen():
        print(f"Serial port {port} opened at 921600 baud.")
    else:
        print("Failed to open serial port.")
        return

    while True:
        try:
            line = ser.readline().decode(errors='ignore').strip()
        except:
            continue

        if not line or "CSI_DATA" not in line:
            if line:
                log_file_fd.write(line + "\n")
            continue

        try:
            csv_reader = csv.reader(StringIO(line))
            row = next(csv_reader)
            
            # Το CSI είναι συνήθως στο τελευταίο στοιχείο της γραμμής
            csi_raw = json.loads(row[-1])
            csi_len = len(csi_raw)

            # Shift buffer (κινεί τα παλιά δεδομένα μια θέση πίσω)
            csi_data_complex[:-1] = csi_data_complex[1:]

            # Μετατροπή των raw τιμών (I, Q) σε μιγαδικούς αριθμούς
            # Στο ESP32-C6 η δομή είναι [imag, real, imag, real...]
            current_frame = np.zeros(CSI_DATA_COLUMNS, dtype=np.complex64)
            idx = 0
            for i in range(0, csi_len, 2):
                if idx >= CSI_DATA_COLUMNS: break
                imag = csi_raw[i]
                real = csi_raw[i+1]
                current_frame[idx] = complex(real, imag)
                idx += 1
            
            csi_data_complex[-1] = current_frame

            # Καταγραφή της γραμμής στο CSV (περιέχει ήδη τα IQ, άρα και τη φάση)
            csv_writer.writerow(row)

        except Exception as e:
            # print(f"Parse error: {e}") # Debugging
            continue


# ==============================
# THREAD
# ==============================
class SerialThread(QThread):
    def __init__(self, port, save_file, log_file):
        super().__init__()
        self.port = port
        self.save_file = save_file
        self.log_file = log_file

    def run(self):
        # Ανοίγουμε τα αρχεία μέσα στο thread
        with open(self.save_file, 'w', newline='') as self.save_fd, \
             open(self.log_file, 'w') as self.log_fd:
            
            self.csv_writer = csv.writer(self.save_fd)
            csi_data_read_parse(self.port, self.csv_writer, self.log_fd)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight CSI Live Viewer")
    parser.add_argument('-p', '--port', required=True, help="Serial port (e.g. COM5)")
    parser.add_argument('-s', '--store', default='csi_data.csv', help="CSV file to store raw CSI")
    parser.add_argument('-l', '--log', default='csi_log.txt', help="Log file for general output")

    args = parser.parse_args()

    app = QApplication(sys.argv)

    # Εκκίνηση του νήματος ανάγνωσης
    thread = SerialThread(args.port, args.store, args.log)
    thread.start()

    # Εμφάνιση παραθύρου
    window = CSIWindow()
    window.show()

    sys.exit(app.exec())
# ESP32-C6 WiFi CSI Communication (Sender & Receiver)

This directory contains the highly optimized, C firmware for ESP32-C6 microcontrollers, designed to generate, transmit, and capture raw Channel State Information (CSI) over Wi-Fi. 

The system consists of two independent ESP-IDF projects: a **Sender** (Transmitter) and a **Receiver** (Sniffer/Logger).

---

## System Architecture

The firmware utilizes **ESP-NOW** for robust, connectionless packet transmission and strictly forces the use of **Wi-Fi 4 (HT40)** on the 2.4 GHz band to capture rich CSI data across 128 subcarriers. 

Both devices are locked to identical parameters (Channel, Transmission Power, Band Mode) to prevent physical layer drift during data collection.

### Core Features (Both Devices)
- **NVS Wear-out Protection:** Detects crash loops (e.g., WDT timeouts) and safely halts execution to prevent permanent Flash memory degradation.
- **External Antenna Configuration:** Forces the use of an external antenna by toggling specific GPIO pins (`GPIO3=LOW`, `GPIO14=HIGH`), ensuring high-gain and directional signal propagation.
- **HT40 Bandwidth Enforcement:** Fixes the ESP-IDF `0x102` error by strictly initializing the 802.11n protocol before setting HT40 bandwidth.
- **Hardware Watchdog (WDT):** Monitors critical tasks to guarantee system stability during extended recording sessions.
- **Fixed TX Power:** Transmission power is statically locked to 18 dBm (`72` in 0.25 dBm units) to eliminate dynamic power scaling artifacts from the CSI amplitude.

---

## 1. The Sender (`csi_send`)

The Sender acts as the active target in the environment, broadcasting packets at a high frequency.

**Key Features:**
- **Deterministic Timing Loop:** Uses `esp_timer_get_time()` and `usleep()` to guarantee a stable transmission frequency (default: 100 Hz), actively compensating for loop execution time drift.
- **Magic Header Payload:** Injects a strict 4-byte signature (`0xA1B2C3D4`) and a sequential ID into every packet. This allows the Receiver to ignore background Wi-Fi noise and only process packets with the specific signature of the sender.
- **Silent & Async ESP-NOW:** The ESP-NOW callback is kept intentionally empty or highly optimized to prevent UART/Print bottlenecks, allowing maximum throughput.
- **Self-Healing Wi-Fi:** Monitors API and Callback failure streaks. If successive errors occur (e.g., due to heavy interference), it completely de-initializes and safely restarts the ESP-NOW interface dynamically.

---

## 2. The Receiver (`csi_recv`)

The Receiver acts as the "Sniffer", capturing the CSI metadata from the physical layer for every packet the Sender broadcasts.

**Key Features:**
- **FreeRTOS Architecture:**
  - **(WiFi Context):** Extremely fast interrupt context. Identifies the Magic Header, captures the AGC/FFT gain, and pushes a struct into a FreeRTOS Queue.
  - **(Print Task):** Pops items from the Queue and performs UART printing (`ets_printf`) safely without blocking the Wi-Fi stack.
- **AGC & FFT Gain Lock:** Records the Automatic Gain Control (AGC) and FFT scaling for the first 100 packets, locks a baseline, and forces the ESP32 hardware to stop dynamically adjusting the gain. It calculates a `compensate_gain` multiplier to guarantee physical amplitude consistency.
- **Static Aligned Buffers:** Uses pre-allocated, RISC-V word-aligned (`__attribute__((aligned(4)))`) buffers for UART strings. Avoids `malloc`/`free` completely to prevent Heap fragmentation and crashes during high-speed logging.
- **Atomic Drop Metrics:** Uses RISC-V atomic variables (`atomic_uint`) to track Queue Overflows or Buffer Truncations lock-free, printing a warning safely in the Consumer task if packets were lost.

---

## Configuration & Flashing

These projects are built using the **Espressif IoT Development Framework (ESP-IDF)** (v5.0+ recommended).

### Prerequisites
1. Install ESP-IDF.
2. Open an ESP-IDF terminal.

### Hardware Setup
In both `csi_send/main/app_main.c` and `csi_recv/main/app_main.c`, ensure the MAC addresses match your specific ESP32-C6 boards:
```c
// Sender
static const uint8_t TARGET_RECEIVER_MAC[] = {0xe4, 0xb3, 0x23, 0xb4, 0x57, 0x7c};

// Receiver
static const uint8_t KNOWN_SENDER_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
```

### Build & Flash (Receiver)
```bash
cd csi_recv
idf.py set-target esp32c6
idf.py build
idf.py -p COM_PORT flash monitor
```
*(Replace `COM_PORT` with your Receiver's port, e.g., `COM3`)*

### Build & Flash (Sender)
```bash
cd ../csi_send
idf.py set-target esp32c6
idf.py build
idf.py -p COM_PORT flash monitor
```
*(Replace `COM_PORT` with your Sender's port, e.g., `COM4`)*

---

## Data Output Format

The Receiver streams the data over USB Serial (UART) in the following strict CSV format, which is parsed by `csi_logger.py` and `csi_parser.py` on the PC:

`type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data`

- **data**: An array of 128 elements representing the raw amplitude of the subcarriers (after AGC compensation).

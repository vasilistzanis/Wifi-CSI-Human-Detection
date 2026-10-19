"""Isolated CSI serial/demo reader and shared-memory snapshots."""

import multiprocessing as mp
import time
from multiprocessing import shared_memory

import numpy as np

import config


def _make_shared_dtype(waveform_len: int, max_subcarriers: int):
    return np.dtype([
        ("n_active",    "i4"),
        ("frame_count", "i8"),
        ("wave_ptr",    "i4"),
        ("variance",    "f8"),
        ("mean_amp",    "f8"),
        ("connected",   "i4"),
        ("fps_n",       "i8"),                  # total samples ever recorded
        ("fps_times",   ("f8", 60)),            # ring buffer of last 60 timestamps
        ("wave_buf",    ("f4", waveform_len)),
        ("last_amp",    ("f4", max_subcarriers)),
    ], align=False)

def _reader_process_fn(shm_name, lock, stop_event, infer_in_q,
                       waveform_len, max_subcarriers,
                       port, baud, demo, rx_buf,
                       window_size, warmup, step, infer_enabled):
    """
    Reader runs entirely in its own OS process. Owns the serial port (or the
    demo generator), writes per-frame state to shared memory, and stacks
    inference windows directly into the inference worker's input queue.

    Because this is a true OS process it has its own GIL — Qt paints, label
    refreshes and even ML inference in the main process cannot starve it.
    """
    import numpy as _np
    import time as _time
    import os as _os
    import math as _math
    import queue as _queue
    import threading as _threading
    from collections import deque as _deque
    from multiprocessing import shared_memory as _shm_mod

    # 1 ms timer resolution + HIGHEST priority on Windows so sleep() and serial
    # blocking reads wake up promptly.
    if _os.name == "nt":
        try:
            import ctypes as _ctypes
            _ctypes.windll.winmm.timeBeginPeriod(1)
            _ctypes.windll.kernel32.SetThreadPriority(
                _ctypes.windll.kernel32.GetCurrentThread(), 2)
        except Exception:
            pass

    shm   = _shm_mod.SharedMemory(name=shm_name)
    dtype = _make_shared_dtype(waveform_len, max_subcarriers)
    state = _np.ndarray((), dtype=dtype, buffer=shm.buf)

    infer_buf_size = window_size + warmup
    infer_deque    = _deque(maxlen=infer_buf_size)
    frames_since   = 0
    forward_queue = None
    forward_failed = _threading.Event()

    def _start_api_forwarder():
        server_url = _os.environ.get("CSI_SERVER_URL", "").strip()
        token = _os.environ.get("CSI_EDGE_TOKEN", "")
        if not server_url and not token:
            return None
        if not server_url or len(token) < 32:
            print(
                "[ERROR] API forwarding disabled: set CSI_SERVER_URL and "
                "CSI_EDGE_TOKEN (32+ characters) together."
            )
            return None

        try:
            from csi_forwarding import send_with_retry, validate_server_url
            validate_server_url(server_url)
        except (ImportError, ValueError) as exc:
            print(f"[ERROR] API forwarding disabled: {exc}")
            return None

        frames = _queue.Queue(maxsize=5000)

        def _send_batches():
            try:
                while not stop_event.is_set() or not frames.empty():
                    try:
                        first = frames.get(timeout=0.25)
                    except _queue.Empty:
                        continue
                    batch = [first]
                    deadline = _time.monotonic() + 0.1
                    while len(batch) < 10 and _time.monotonic() < deadline:
                        try:
                            batch.append(
                                frames.get(timeout=max(0.01, deadline - _time.monotonic()))
                            )
                        except _queue.Empty:
                            break
                    accepted = send_with_retry(server_url, token, batch)
                    if accepted != len(batch):
                        print(f"[ERROR] API accepted {accepted} of {len(batch)} CSI frames.")
                    else:
                        print(f"Forwarded {accepted} CSI frames to API.")
            except (RuntimeError, ValueError, OSError) as exc:
                forward_failed.set()
                print(f"[ERROR] API forwarding stopped: {exc}")

        sender = _threading.Thread(
            target=_send_batches,
            name="csi-api-forwarder",
            daemon=True,
        )
        sender.start()
        print(f"[OK]  API forwarding enabled: {server_url}")
        return frames

    def _push(cf_frame):
        nonlocal frames_since
        n = min(int(cf_frame.size), max_subcarriers)
        cf = cf_frame[:n].astype(_np.complex64, copy=False)
        amp = _np.abs(cf)
        ma  = float(amp.mean())
        mx  = float(amp.max())
        amp_norm = amp / mx if mx > 0 else amp
        now = _time.monotonic()

        with lock:
            p = int(state["wave_ptr"])
            state["wave_buf"][p]  = ma
            state["wave_ptr"]     = (p + 1) % waveform_len
            state["last_amp"][:n] = amp_norm
            if n > int(state["n_active"]):
                state["n_active"] = n
            state["frame_count"] += 1

            fps_n = int(state["fps_n"])
            state["fps_times"][fps_n % 60] = now
            state["fps_n"] = fps_n + 1

            prev_var = float(state["variance"])
            prev_ma  = float(state["mean_amp"])
            state["variance"] = prev_var * 0.97 + (ma - prev_ma) ** 2 * 0.03
            state["mean_amp"] = prev_ma  * 0.97 + ma * 0.03
            fc = int(state["frame_count"])

        if infer_enabled:
            infer_deque.append(cf_frame)
            frames_since += 1
            if frames_since >= step and len(infer_deque) >= window_size:
                frames_since = 0
                try:
                    cm = _np.vstack(list(infer_deque))
                    try:
                        infer_in_q.get_nowait()
                    except Exception:
                        pass
                    infer_in_q.put_nowait((cm, fc))
                except Exception:
                    pass

    def _run_demo():
        sc       = max_subcarriers
        rng      = _np.random.default_rng(0)
        t        = 0.0
        idx      = _np.arange(sc, dtype=_np.float32)
        idx_norm = idx / sc
        interval = 1.0 / 100.0
        next_t   = _time.perf_counter()
        with lock:
            state["connected"] = 1
        while not stop_event.is_set():
            next_t   += interval
            remaining = next_t - _time.perf_counter()
            if remaining > 0.001:
                _time.sleep(remaining)
            t  += 0.05
            p1  = _np.exp(-0.5 * ((idx - (sc * .3 + 10 * _math.sin(t * .7))) / 6) ** 2)
            p2  = _np.exp(-0.5 * ((idx - (sc * .7 +  8 * _math.cos(t * .5))) / 5) ** 2)
            amp = p1 * .7 + p2 * .5 + rng.uniform(0, .04, sc).astype(_np.float32)
            ph  = t * .8 + idx_norm * (2 * _math.pi)
            frame = (amp * (_np.cos(ph) + 1j * _np.sin(ph))).astype(_np.complex64)
            _push(frame)

    def _run_serial():
        nonlocal forward_queue
        try:
            import serial
            from csi_parser import parse_csi_line as _parse
            from csi_serial import iter_csi_lines
        except ImportError:
            print("[ERROR] pyserial / csi_parser missing → demo mode")
            _run_demo()
            return
        ser = None
        dropped_api_frames = 0
        try:
            ser = serial.Serial(port, baud, timeout=0.5)
            if _os.name == "nt" and hasattr(ser, "set_buffer_size"):
                ser.set_buffer_size(rx_size=rx_buf)
            ser.reset_input_buffer()
            with lock:
                state["connected"] = 1
            print(f"[OK]  Connected: {port} @ {baud}")
            forward_queue = _start_api_forwarder()
        except Exception as e:
            print(f"[ERROR] {e}  → demo mode")
            _run_demo()
            return
        try:
            for line in iter_csi_lines(ser, stop_event, decode_errors="ignore"):
                if forward_queue is not None and not forward_failed.is_set():
                    try:
                        forward_queue.put_nowait(line)
                    except _queue.Full:
                        dropped_api_frames += 1
                        if dropped_api_frames == 1 or dropped_api_frames % 100 == 0:
                            print(
                                "[ERROR] API forwarding queue is full; "
                                f"{dropped_api_frames} CSI frames dropped from API forwarding."
                            )
                frame = _parse(line)
                if frame is not None:
                    _push(frame)
        finally:
            if ser and ser.is_open:
                ser.close()
            with lock:
                state["connected"] = 0

    try:
        if demo:
            _run_demo()
        else:
            _run_serial()
    finally:
        try:
            shm.close()
        except Exception:
            pass

class ReaderProcess:
    """
    GUI-side handle for the reader child process. Exposes the same snapshot()
    shape as the old ReaderThread so the existing pages and refresh loop work
    unchanged. State is read via a structured numpy view over shared memory.
    """

    def __init__(self, port, baud, demo, waveform_len, rx_buf,
                 window_size, warmup, step, infer_enabled=True):
        self.port          = port
        self.baud          = baud
        self.demo          = demo
        self.waveform_len  = waveform_len
        self.rx_buf        = rx_buf
        self._window_size  = window_size
        self._warmup       = warmup
        self._step         = step
        self._infer_enabled = infer_enabled
        self._max_sc       = config.MAX_SUBCARRIERS
        self.infer_buf_max = window_size + warmup
        self._start_time   = time.monotonic()

        self._dtype = _make_shared_dtype(waveform_len, self._max_sc)
        size = self._dtype.itemsize
        self._shm = shared_memory.SharedMemory(create=True, size=size)
        # Zero-init the block so the structured view reads sane defaults.
        self._shm.buf[:size] = b"\x00" * size
        self._state = np.ndarray((), dtype=self._dtype, buffer=self._shm.buf)

        self._lock        = mp.Lock()
        self._stop_event  = mp.Event()
        self._infer_in_q  = mp.Queue(maxsize=2)
        self._proc        = None

    @property
    def infer_in_q(self):
        """The inference input queue — handed to InferenceProcess so the reader
        can push windows directly without going through the GUI thread."""
        return self._infer_in_q

    @property
    def connected(self) -> bool:
        with self._lock:
            return bool(int(self._state["connected"]))

    def start(self):
        self._proc = mp.Process(
            target=_reader_process_fn,
            args=(self._shm.name, self._lock, self._stop_event, self._infer_in_q,
                  self.waveform_len, self._max_sc,
                  self.port, self.baud, self.demo, self.rx_buf,
                  self._window_size, self._warmup, self._step,
                  self._infer_enabled),
            daemon=True,
        )
        self._proc.start()

    def snapshot(self) -> dict | None:
        with self._lock:
            n = int(self._state["n_active"])
            if n == 0:
                return None
            fc       = int(self._state["frame_count"])
            wave_ptr = int(self._state["wave_ptr"])
            wave_buf = self._state["wave_buf"].copy()
            amp_snap = self._state["last_amp"][:n].copy()
            fps_n    = int(self._state["fps_n"])
            ft_buf   = self._state["fps_times"].copy()
            var      = float(self._state["variance"])
            ma       = float(self._state["mean_amp"])
            conn     = bool(int(self._state["connected"]))

        # FPS from the most recent min(fps_n, 60) timestamps in the ring.
        if fps_n >= 2:
            n_valid   = min(fps_n, 60)
            last_idx  = (fps_n - 1) % 60
            first_idx = (fps_n - n_valid) % 60
            ft_last   = float(ft_buf[last_idx])
            ft_first  = float(ft_buf[first_idx])
            fps = (n_valid - 1) / (ft_last - ft_first) if ft_last > ft_first else 0.0
        else:
            fps = 0.0

        return {
            "wave":        np.roll(wave_buf, -wave_ptr),
            "last_amp":    amp_snap,
            "n":           n,
            "frame_count": fc,
            "fps":         fps,
            "uptime_s":    time.monotonic() - self._start_time,
            "variance":    var,
            "mean_amp":    ma,
            # Reader owns the inference window now — we report fullness as the
            # ratio of "ready" (have window_size frames) to keep the existing
            # sidebar buffer bar meaningful.
            "infer_fill":  min(1.0, fc / max(1, self._window_size)),
            "connected":   conn,
        }

    def stop(self):
        self._stop_event.set()
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass

    def join(self, timeout=2.0):
        if self._proc is not None:
            self._proc.join(timeout=timeout)

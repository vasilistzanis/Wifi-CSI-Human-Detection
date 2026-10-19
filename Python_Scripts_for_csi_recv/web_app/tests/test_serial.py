from __future__ import annotations

from collections import deque
from threading import Event

from csi_serial import iter_csi_lines


class FakeSerial:
    def __init__(self, lines: list[bytes], stopped: Event) -> None:
        self.lines = deque(lines)
        self.stopped = stopped

    def readline(self) -> bytes:
        if self.lines:
            return self.lines.popleft()
        self.stopped.set()
        return b""


def test_iter_csi_lines_filters_non_csi_and_decodes_invalid_bytes() -> None:
    stopped = Event()
    serial_port = FakeSerial(
        [b"boot message\n", b"CSI_DATA,1,\xff\n"],
        stopped,
    )

    assert list(iter_csi_lines(serial_port, stopped)) == ["CSI_DATA,1,\ufffd"]


def test_iter_csi_lines_can_preserve_dashboard_decode_policy() -> None:
    stopped = Event()
    serial_port = FakeSerial([b"CSI_DATA,1,\xff\n"], stopped)

    assert list(iter_csi_lines(serial_port, stopped, decode_errors="ignore")) == [
        "CSI_DATA,1,"
    ]

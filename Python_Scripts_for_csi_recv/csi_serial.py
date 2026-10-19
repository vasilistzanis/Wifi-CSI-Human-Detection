from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, Protocol


class SerialReader(Protocol):
    def readline(self) -> bytes: ...


class StopSignal(Protocol):
    def is_set(self) -> bool: ...


def iter_csi_lines(
    serial_port: SerialReader,
    stopped: StopSignal,
    *,
    decode_errors: Literal["replace", "ignore"] = "replace",
) -> Iterator[str]:
    while not stopped.is_set():
        raw_line = serial_port.readline()
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors=decode_errors).strip()
        if line.startswith("CSI_DATA"):
            yield line

from __future__ import annotations

from collections import deque

from apps.desktop.desktop_inference import InferenceProcess, inference_worker


class FakeInputQueue:
    def __init__(self, messages: list[object]) -> None:
        self.messages = deque(messages)

    def get(self, timeout: float) -> object:
        return self.messages.popleft()


class FakeOutputQueue:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def put_nowait(self, message: object) -> None:
        self.messages.append(message)


def test_worker_forwards_empty_prediction_for_unavailable_pipeline() -> None:
    input_queue = FakeInputQueue([("unused", 12), None])
    output_queue = FakeOutputQueue()

    inference_worker(
        input_queue,
        output_queue,
        pipeline=None,
        model=None,
        label_encoder=None,
        window_size=10,
        cutoff=8.5,
    )

    assert output_queue.messages == [(None, None, None, 0.0, 12)]


def test_inference_process_keeps_dashboard_label_encoder_keyword() -> None:
    label_encoder = object()

    process = InferenceProcess(
        in_q=object(),
        pipeline=None,
        model=None,
        le=label_encoder,
        classes=[],
        window_size=10,
        cutoff=8.5,
    )

    assert process._label_encoder is label_encoder

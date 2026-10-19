from __future__ import annotations

import multiprocessing as mp


def inference_worker(
    in_q,
    out_q,
    pipeline,
    model,
    label_encoder,
    window_size,
    cutoff,
) -> None:
    """Run inference in a child process so model work cannot block the UI."""
    import numpy as np
    import time

    try:
        from csi_ml_pipeline import extract_features_from_window
    except ImportError:
        extract_features_from_window = None

    try:
        from csi_ml_pipeline import N_STATS
    except ImportError:
        N_STATS = 25
    n_pca = None
    if pipeline is not None and hasattr(pipeline, "pca") and pipeline.pca is not None:
        n_pca = getattr(pipeline.pca, "n_components_", None)
    if n_pca and hasattr(model, "n_features_in_"):
        n_stats = model.n_features_in_ // n_pca
    else:
        n_stats = N_STATS

    mismatch_warned = False
    while True:
        try:
            message = in_q.get(timeout=1.0)
        except Exception:
            continue
        if message is None:
            break

        csi_matrix, frame_count = message
        raw_candidate = confidence_candidate = probabilities_candidate = None
        latency = 0.0

        if pipeline is not None and extract_features_from_window is not None:
            if csi_matrix.shape[0] >= window_size + 1:
                try:
                    if csi_matrix.shape[1] != pipeline._fitted_n_subcarriers:
                        if not mismatch_warned:
                            import sys

                            print(
                                "[WARN] Subcarrier mismatch in inference: "
                                f"got {csi_matrix.shape[1]}, "
                                f"expected {pipeline._fitted_n_subcarriers}. "
                                "Check hardware config (bandwidth/channel).",
                                file=sys.stderr,
                                flush=True,
                            )
                            mismatch_warned = True
                    else:
                        started_at = time.monotonic()
                        processed = pipeline.transform(
                            csi_matrix,
                            use_pca=True,
                            cutoff=cutoff,
                        )
                        if processed.shape[0] >= window_size:
                            features = extract_features_from_window(
                                processed[-window_size:],
                                fs=pipeline.fs,
                                cutoff_hz=cutoff,
                                n_stats=n_stats,
                            ).reshape(1, -1)
                            if np.all(np.isfinite(features)):
                                valid_features = True
                                if hasattr(model, "n_features_in_"):
                                    valid_features = (
                                        features.shape[1] == model.n_features_in_
                                    )
                                    if not valid_features:
                                        import sys

                                        print(
                                            "[WARN] Feature count mismatch: model expects "
                                            f"{model.n_features_in_}, got {features.shape[1]}. "
                                            "Retrain the model.",
                                            file=sys.stderr,
                                            flush=True,
                                        )
                                if valid_features:
                                    if hasattr(model, "predict_proba"):
                                        probabilities = model.predict_proba(features)[0]
                                        index = int(np.argmax(probabilities))
                                        confidence_candidate = (
                                            float(probabilities[index]) * 100.0
                                        )
                                    else:
                                        index = int(model.predict(features)[0])
                                        probabilities = np.zeros(
                                            len(label_encoder.classes_),
                                            dtype=np.float32,
                                        )
                                        probabilities[index] = 1.0
                                        confidence_candidate = 100.0
                                    raw_candidate = str(
                                        label_encoder.inverse_transform([index])[0]
                                    )
                                    probabilities_candidate = probabilities
                        latency = (time.monotonic() - started_at) * 1000.0
                except Exception:
                    pass

        try:
            out_q.put_nowait(
                (
                    raw_candidate,
                    confidence_candidate,
                    probabilities_candidate,
                    latency,
                    frame_count,
                )
            )
        except Exception:
            pass


class InferenceProcess:
    """Manage the child process that performs desktop CSI inference."""

    def __init__(
        self,
        in_q,
        pipeline,
        model,
        le,
        classes,
        window_size,
        cutoff,
    ) -> None:
        self._in_q = in_q
        self._pipeline = pipeline
        self._model = model
        self._label_encoder = le
        self._classes = list(classes)
        self._window_size = window_size
        self._cutoff = cutoff
        self._out_q = None
        self._proc = None

    def start(self) -> None:
        self._out_q = mp.Queue(maxsize=8)
        self._proc = mp.Process(
            target=inference_worker,
            args=(
                self._in_q,
                self._out_q,
                self._pipeline,
                self._model,
                self._label_encoder,
                self._window_size,
                self._cutoff,
            ),
            daemon=True,
        )
        self._proc.start()

    def get_result(self):
        if self._out_q is None:
            return None
        result = None
        while True:
            try:
                result = self._out_q.get_nowait()
            except Exception:
                break
        return result

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._in_q.put(None, timeout=2.0)
        except Exception:
            pass
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()
        try:
            self._out_q.close()
        except Exception:
            pass

    def restart(self, pipeline, model, label_encoder, classes) -> None:
        self.stop()
        self._pipeline = pipeline
        self._model = model
        self._label_encoder = label_encoder
        self._classes = list(classes)
        self.start()

    def join(self, timeout=2.0) -> None:
        self.stop()

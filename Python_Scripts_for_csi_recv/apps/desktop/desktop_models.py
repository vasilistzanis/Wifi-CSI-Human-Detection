"""Model artifact loading for the desktop inference runtime."""

from pathlib import Path


def load_models(models_dir: str, model_key: str):
    directory = Path(models_dir)
    paths = {
        "pipeline": directory / "csi_pipeline.joblib",
        "le": directory / "label_encoder.joblib",
        "model": directory / f"{model_key}.joblib",
    }
    for artifact in paths.values():
        if not artifact.exists():
            raise FileNotFoundError(
                f"Missing: {artifact}\n  Train first: python csi_ml_pipeline.py --save_model"
            )

    import joblib

    return (
        joblib.load(paths["pipeline"]),
        joblib.load(paths["le"]),
        joblib.load(paths["model"]),
    )

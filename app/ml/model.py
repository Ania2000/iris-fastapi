from pathlib import Path
from typing import Any

import joblib
import numpy as np

MODEL_PATH = Path("models/iris_model.joblib")


class ModelNotLoadedError(RuntimeError):
    pass


class IrisModel:
    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run training: uv run python scripts/train.py"
            )
        self._bundle = joblib.load(MODEL_PATH)

    @property
    def target_names(self) -> list[str]:
        if self._bundle is None:
            raise ModelNotLoadedError("Model not loaded")
        return list(self._bundle["target_names"])

    def predict(self, features: list[float]) -> tuple[int, str]:
        if self._bundle is None:
            raise ModelNotLoadedError("Model not loaded")

        model = self._bundle["model"]
        X = np.array([features], dtype=float)
        pred_class: int = int(model.predict(X)[0])
        pred_label = self.target_names[pred_class]
        return pred_class, pred_label
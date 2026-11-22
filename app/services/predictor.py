import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from app.schemas.clinical import FEATURE_COLUMNS, HematologyInput, PATHOLOGY_COLS

logger = logging.getLogger("uvicorn.error")

class HematologyPredictor:
    def __init__(self, model_path: str = "models/multilabel_model.joblib") -> None:
        """Loads the trained multi-label model once at startup."""
        logger.info(f"Loading hematology model from {model_path}...")
        self.model_path = model_path
        resolved_path = Path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model = joblib.load(resolved_path)
        logger.info("Hematology model loaded successfully!")

    def _to_dataframe(self, payload: HematologyInput) -> pd.DataFrame:
        """Builds a 1-row DataFrame using only the 16 features in exact training order."""
        row = {col: getattr(payload, col) for col in FEATURE_COLUMNS}
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def _format_predictions(self, raw_pred: np.ndarray) -> dict[str, int]:
        """Converts MultiOutputClassifier output into a {PATHOLOGY: 0_or_1} dict."""
        arr = np.asarray(raw_pred)
        if arr.ndim == 2 and arr.shape[1] == len(PATHOLOGY_COLS):
            values = [int(round(float(v))) for v in arr[0]]
        elif arr.ndim == 1 and arr.shape[0] == len(PATHOLOGY_COLS):
            values = [int(round(float(v))) for v in arr]
        else:
            raise ValueError(
                f"Expected output with {len(PATHOLOGY_COLS)} targets, got shape {arr.shape}"
            )
        return dict(zip(PATHOLOGY_COLS, values))

    def _extract_probabilities(self, features_df: pd.DataFrame) -> dict[str, float] | None:
        """Extracts positive-class probabilities from MultiOutputClassifier.predict_proba."""
        if not hasattr(self.model, "predict_proba"):
            return None

        try:
            probas = self.model.predict_proba(features_df)
        except Exception:
            return None

        if isinstance(probas, list) and len(probas) == len(PATHOLOGY_COLS):
            result: dict[str, float] = {}
            for col_name, class_probs in zip(PATHOLOGY_COLS, probas):
                arr = np.asarray(class_probs)
                prob_one = float(arr[0, 1]) if arr.ndim == 2 and arr.shape[1] > 1 else float(arr[0, 0])
                result[col_name] = round(prob_one, 4)
            return result

        return None

    def predict(self, payload: HematologyInput) -> dict:
        """Runs inference on a single patient payload and returns predictions."""
        features_df = self._to_dataframe(payload)
        raw_pred = self.model.predict(features_df)
        predictions = self._format_predictions(raw_pred)
        probabilities = self._extract_probabilities(features_df)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
        }
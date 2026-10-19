"""Load the trained detection model and score a feature vector.

The model artifact (selected by PR-AUC, the project's primary metric for this
imbalanced problem) is loaded once at import. `recommended_action` and the flag
threshold are the notebook's own, so the served verdict matches the notebook's.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.qm.features import MODEL_FEATURES

__all__ = ["score_feature_vector", "recommended_action", "MODEL_INFO"]

_ARTIFACT_PATH = Path(__file__).parent / "models" / "quote_risk_model.joblib"
_ARTIFACT = joblib.load(_ARTIFACT_PATH)
_MODEL = _ARTIFACT["model"]
_FEATURES = _ARTIFACT["features"]
_FLAG_THRESHOLD = _ARTIFACT.get("flag_threshold", 0.5)

assert _FEATURES == MODEL_FEATURES, "model feature order differs from features.py"

# Small, non-sensitive summary the API can expose.
MODEL_INFO = {
    "model_name": _ARTIFACT.get("model_name"),
    "selection_metric": _ARTIFACT.get("selection_metric"),
    "flag_threshold": _FLAG_THRESHOLD,
    "n_features": len(_FEATURES),
    "sklearn_version": _ARTIFACT.get("sklearn_version"),
}


def recommended_action(prob: float) -> str:
    """Notebook's action bands (cell 6), verbatim."""
    if prob < 0.30:
        return "No action"
    if prob < 0.60:
        return "Prompt customer to confirm details"
    if prob < 0.80:
        return "Request additional validation"
    return "Escalate for manual review"


def score_feature_vector(feature_vector: Dict[str, Any]) -> Dict[str, Any]:
    """Score a 31-feature vector. Returns risk_score, flag, and recommended_action."""
    missing = [f for f in _FEATURES if f not in feature_vector]
    if missing:
        raise ValueError(f"missing features: {missing}")

    X = pd.DataFrame([[feature_vector[f] for f in _FEATURES]], columns=_FEATURES)
    risk_score = float(_MODEL.predict_proba(X)[:, 1][0])

    return {
        "risk_score": round(risk_score, 3),
        "flag": bool(risk_score >= _FLAG_THRESHOLD),
        "recommended_action": recommended_action(risk_score),
    }

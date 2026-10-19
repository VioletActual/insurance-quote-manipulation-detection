import pytest

from src.qm.features import journey_to_features
from src.qm.scoring import score_feature_vector, recommended_action, MODEL_INFO


def _q(age, exp, acc, mileage, premium, t):
    return {
        "driver_age": age, "driver_experience": exp, "previous_accidents": acc,
        "annual_mileage": mileage, "car_age": 5, "premium": premium, "quote_time": t,
    }


SUSPICIOUS = [
    _q(24, 2, 2, 18.0, 1300, "2026-03-01T09:00:00"),
    _q(24, 3, 1, 14.0, 1100, "2026-03-01T09:12:00"),
    _q(25, 4, 0, 10.0, 820, "2026-03-01T09:21:00"),
    _q(25, 5, 0, 8.0, 700, "2026-03-01T09:28:00"),
]
BENIGN = [_q(45, 20, 0, 9.0, 560, "2026-03-01T09:00:00")]


def test_model_info_is_random_forest():
    assert MODEL_INFO["model_name"] == "random_forest"
    assert MODEL_INFO["selection_metric"] == "pr_auc"
    assert MODEL_INFO["n_features"] == 31


def test_recommended_action_bands():
    assert recommended_action(0.1) == "No action"
    assert recommended_action(0.4) == "Prompt customer to confirm details"
    assert recommended_action(0.7) == "Request additional validation"
    assert recommended_action(0.9) == "Escalate for manual review"


def test_score_shape_and_range():
    res = score_feature_vector(journey_to_features(BENIGN)["feature_vector"])
    assert 0.0 <= res["risk_score"] <= 1.0
    assert isinstance(res["flag"], bool)
    assert isinstance(res["recommended_action"], str)


def test_suspicious_scores_higher_than_benign():
    hi = score_feature_vector(journey_to_features(SUSPICIOUS)["feature_vector"])
    lo = score_feature_vector(journey_to_features(BENIGN)["feature_vector"])
    assert hi["risk_score"] > lo["risk_score"]
    assert hi["flag"] is True


def test_scoring_is_deterministic():
    fv = journey_to_features(SUSPICIOUS)["feature_vector"]
    assert score_feature_vector(fv)["risk_score"] == score_feature_vector(fv)["risk_score"]


def test_missing_feature_raises():
    with pytest.raises(ValueError):
        score_feature_vector({"n_quotes": 3})

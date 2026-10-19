from fastapi.testclient import TestClient

from src.qm.api.main import app

client = TestClient(app)


def _q(age, exp, acc, mileage, premium, t):
    return {
        "driver_age": age, "driver_experience": exp, "previous_accidents": acc,
        "annual_mileage": mileage, "car_age": 5, "premium": premium, "quote_time": t,
    }


SUSPICIOUS = {
    "customer_id": 662,
    "quotes": [
        _q(24, 2, 2, 18.0, 1300, "2026-03-01T09:00:00"),
        _q(24, 3, 1, 14.0, 1100, "2026-03-01T09:12:00"),
        _q(25, 4, 0, 10.0, 820, "2026-03-01T09:21:00"),
        _q(25, 5, 0, 8.0, 700, "2026-03-01T09:28:00"),
    ],
}
BENIGN = {"customer_id": 101, "quotes": [_q(45, 20, 0, 9.0, 560, "2026-03-01T09:00:00")]}


def test_model_info_endpoint():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["model_name"] == "random_forest"


def test_score_flags_suspicious_journey():
    r = client.post("/score", json=SUSPICIOUS)
    assert r.status_code == 200
    body = r.json()
    assert body["flag"] is True
    assert body["risk_score"] >= 0.5
    assert body["recommended_action"]
    assert body["model_name"] == "random_forest"
    # The returned evidence is schema-valid and carries the journey.
    assert body["case_evidence"]["customer_id"] == 662
    assert body["case_evidence"]["n_quotes"] == 4


def test_score_benign_journey_not_flagged():
    r = client.post("/score", json=BENIGN)
    assert r.status_code == 200
    assert r.json()["flag"] is False


def test_score_rejects_empty_journey():
    r = client.post("/score", json={"customer_id": 1, "quotes": []})
    assert r.status_code == 422


def test_score_rejects_bad_quote():
    bad = {"customer_id": 1, "quotes": [_q(24, 2, 2, 18.0, 0, "2026-03-01T09:00:00")]}  # premium must be > 0
    r = client.post("/score", json=bad)
    assert r.status_code == 422


def test_score_rejects_invalid_timestamp():
    # The Swagger default fills quote_time with "string"; that must 422, not 500.
    bad = {"customer_id": 1, "quotes": [_q(24, 2, 2, 18.0, 500, "string")]}
    r = client.post("/score", json=bad)
    assert r.status_code == 422


def test_score_evidence_chains_into_explain():
    # The full pipeline: /score -> take case_evidence -> /explain.
    scored = client.post("/score", json=SUSPICIOUS).json()
    evidence = scored["case_evidence"]
    explained = client.post("/explain", json=evidence)
    assert explained.status_code == 200
    body = explained.json()
    assert len(body["explanation"]) > 0
    assert body["scores"]["groundedness"] == 1
    assert body["scores"]["factual_correctness"] == 1

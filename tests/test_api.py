from fastapi.testclient import TestClient

from src.qm.api.main import app

client = TestClient(app)


# A realistic flagged case, schema-valid (mirrors a high-risk referral row).
SUSPICIOUS_EVIDENCE = {
    "customer_id": 662,
    "risk_score": 0.907,
    "recommended_action": "Escalate for manual review",
    "n_quotes": 5,
    "first_to_lowest_premium_drop_pct": 31.59,
    "largest_single_drop_pct": 16.4,
    "material_edit_count": 6,
    "suspicious_field_edit_count": 7,
    "suspicious_step_ratio": 1.0,
    "mileage_decrease_count": 2,
    "accident_decrease_count": 1,
    "experience_increase_count": 1,
    "age_band_cross_count": 1,
    "consistency_violation_count": 1,
    "quote_sequence": [
        {"quote_number": 1, "premium": 980.0},
        {"quote_number": 5, "premium": 670.5},
    ],
}

# A low-signal case: nothing crosses the stub's thresholds.
QUIET_EVIDENCE = {
    "customer_id": 101,
    "risk_score": 0.12,
    "recommended_action": "No action required",
    "n_quotes": 1,
    "first_to_lowest_premium_drop_pct": 0.0,
    "largest_single_drop_pct": 0.0,
    "material_edit_count": 0,
    "suspicious_field_edit_count": 0,
    "suspicious_step_ratio": 0.0,
    "mileage_decrease_count": 0,
    "accident_decrease_count": 0,
    "experience_increase_count": 0,
    "age_band_cross_count": 0,
    "consistency_violation_count": 0,
    "quote_sequence": [{"quote_number": 1, "premium": 540.0}],
}


# --- health ---

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# --- explain ---

def test_explain_returns_grounded_scored_explanation():
    r = client.post("/explain", json=SUSPICIOUS_EVIDENCE)
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == 662
    assert body["llm_mode"] == "stub"
    assert len(body["explanation"]) > 0
    # The deterministic generator must never accuse, and must stay grounded.
    assert body["scores"]["groundedness"] == 1
    assert body["scores"]["factual_correctness"] == 1
    assert body["scores"]["conciseness"] == 1


def test_explain_quiet_case_falls_back_to_generic_reason():
    r = client.post("/explain", json=QUIET_EVIDENCE)
    assert r.status_code == 200
    body = r.json()
    assert "elevated risk score" in body["explanation"]
    assert body["scores"]["groundedness"] == 1


def test_explain_rejects_out_of_range_risk_score():
    bad = dict(SUSPICIOUS_EVIDENCE, risk_score=1.5)
    r = client.post("/explain", json=bad)
    assert r.status_code == 422  # pydantic validation, caught at the boundary


def test_explain_rejects_missing_field():
    bad = {k: v for k, v in SUSPICIOUS_EVIDENCE.items() if k != "n_quotes"}
    r = client.post("/explain", json=bad)
    assert r.status_code == 422


def test_explain_rejects_empty_quote_sequence():
    bad = dict(SUSPICIOUS_EVIDENCE, quote_sequence=[])
    r = client.post("/explain", json=bad)
    assert r.status_code == 422


# --- evaluate ---

def test_evaluate_clean_explanation_passes_gate():
    payload = {
        "explanation": "This journey may warrant review based on 5 quote attempts.",
        "case_evidence": SUSPICIOUS_EVIDENCE,
    }
    r = client.post("/evaluate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["groundedness"] == 1
    assert body["factual_correctness"] == 1
    assert body["passed_gate"] is True
    assert body["format_compliance"] is None  # no prompt_version supplied


def test_evaluate_accusatory_text_fails_gate():
    payload = {
        "explanation": "The customer was deliberately manipulating the quote.",
        "case_evidence": SUSPICIOUS_EVIDENCE,
    }
    r = client.post("/evaluate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["groundedness"] == 0
    assert body["passed_gate"] is False


def test_evaluate_hallucinated_number_fails_factual():
    payload = {
        "explanation": "The customer made 42 quote attempts.",
        "case_evidence": SUSPICIOUS_EVIDENCE,
    }
    r = client.post("/evaluate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["factual_correctness"] == 0
    assert body["passed_gate"] is False


def test_evaluate_format_compliance_runs_for_v3():
    payload = {
        "explanation": "REASONS:\n- Five quotes were run.\n- Mileage fell.\nACTION: Escalate for manual review.",
        "case_evidence": SUSPICIOUS_EVIDENCE,
        "prompt_version": "v3_structured",
    }
    r = client.post("/evaluate", json=payload)
    assert r.status_code == 200
    assert r.json()["format_compliance"] == 1


def test_evaluate_generated_explanation_is_self_consistent():
    # Explain then evaluate the same text: the gate it reports should agree.
    gen = client.post("/explain", json=SUSPICIOUS_EVIDENCE).json()["explanation"]
    r = client.post("/evaluate", json={"explanation": gen, "case_evidence": SUSPICIOUS_EVIDENCE})
    assert r.json()["passed_gate"] is True

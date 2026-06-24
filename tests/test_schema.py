import pytest
from src.qm.schema import (
    CaseEvidence, StructuredExplanation,
    parse_structured_explanation, ValidationError,
)

VALID_EVIDENCE = dict(
    customer_id=662, risk_score=0.907,
    recommended_action="Escalate for manual review",
    n_quotes=5, first_to_lowest_premium_drop_pct=31.59,
    largest_single_drop_pct=16.4, material_edit_count=7,
    suspicious_field_edit_count=7, suspicious_step_ratio=1.0,
    mileage_decrease_count=1, accident_decrease_count=1,
    experience_increase_count=1, age_band_cross_count=0,
    consistency_violation_count=0,
    quote_sequence=[{"quote_number": 1, "premium": 500.0}],
)

GOOD_OUTPUT = (
    "REASONS:\n"
    "- The customer received 5 quotes with a 31.59% premium drop.\n"
    "- 7 suspicious field edits were made across the journey.\n"
    "ACTION: Escalate for manual review"
)

def test_valid_evidence_passes():
    CaseEvidence(**VALID_EVIDENCE)

def test_risk_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        CaseEvidence(**{**VALID_EVIDENCE, "risk_score": 1.5})

def test_structured_output_parses():
    obj = parse_structured_explanation(GOOD_OUTPUT)
    assert len(obj.reasons) == 2
    assert obj.action == "Escalate for manual review"

def test_too_many_reasons_rejected():
    with pytest.raises(ValidationError):
        StructuredExplanation(reasons=["a", "b", "c", "d"], action="review")

def test_overlong_reason_rejected():
    with pytest.raises(ValidationError):
        StructuredExplanation(reasons=["word " * 21], action="review")

def test_unstructured_text_rejected():
    with pytest.raises(ValidationError):
        parse_structured_explanation("This was flagged because 5 quotes were made.")
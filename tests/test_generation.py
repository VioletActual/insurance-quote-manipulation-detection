from src.qm.generation import generate_explanation, generate_referral_explanation_stub
from src.qm.evaluation import score_groundedness

EVIDENCE = {
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
    "quote_sequence": [{"quote_number": 1, "premium": 980.0}],
}


def test_generation_is_deterministic():
    assert generate_explanation(EVIDENCE) == generate_explanation(EVIDENCE)


def test_generation_mentions_recommended_action():
    assert "Escalate for manual review" in generate_explanation(EVIDENCE)


def test_generation_passes_groundedness_scorer():
    # Uses the project's real groundedness rubric, not a hand-rolled word list,
    # so a legitimate disclaimer like "does not prove fraud" is not flagged.
    assert score_groundedness(generate_explanation(EVIDENCE)) == 1


def test_generation_falls_back_when_no_signals():
    quiet = dict(EVIDENCE, n_quotes=1, first_to_lowest_premium_drop_pct=0.0,
                 material_edit_count=0, mileage_decrease_count=0, accident_decrease_count=0,
                 experience_increase_count=0, age_band_cross_count=0, consistency_violation_count=0)
    assert "elevated risk score" in generate_referral_explanation_stub(quiet)

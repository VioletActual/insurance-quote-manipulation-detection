"""Deterministic referral-explanation generation.

Lifted out of the V3 notebook so it can be imported, tested, and served.
This is the key-free fallback path: it never calls an external model, so the
API, the test suite, and CI all run with no API key and no network.

The live LLM path (Groq / OpenAI-compatible) stays in the notebook; the served
API uses this deterministic generator only.
"""
from __future__ import annotations

from typing import Any, Dict

__all__ = ["generate_referral_explanation_stub", "generate_explanation"]


def generate_referral_explanation_stub(case_evidence: Dict[str, Any]) -> str:
    """Deterministic fallback. Also used in CI so tests never call an external API.

    Takes a CaseEvidence dict (e.g. CaseEvidence(...).model_dump()) and returns a
    plain-English referral explanation grounded only in the supplied evidence.
    """
    reasons = []
    if case_evidence["n_quotes"] >= 4:
        reasons.append(f"{case_evidence['n_quotes']} quote attempts were made")
    if case_evidence["first_to_lowest_premium_drop_pct"] >= 10:
        reasons.append(
            f"the premium reduced by {case_evidence['first_to_lowest_premium_drop_pct']}% from first to lowest quote"
        )
    if case_evidence["material_edit_count"] > 0:
        reasons.append(f"{case_evidence['material_edit_count']} material rating-factor edits were observed")
    if case_evidence["mileage_decrease_count"] > 0:
        reasons.append("declared mileage decreased during the journey")
    if case_evidence["accident_decrease_count"] > 0:
        reasons.append("previous accident count decreased during the journey")
    if case_evidence["experience_increase_count"] > 0:
        reasons.append("declared driving experience increased during the journey")
    if case_evidence["age_band_cross_count"] > 0:
        reasons.append("an age-band threshold crossing occurred")
    if case_evidence["consistency_violation_count"] > 0:
        reasons.append("age/experience consistency issues were detected")
    if not reasons:
        reasons.append("the model assigned an elevated risk score based on combined behavioural signals")

    return (
        f"This journey was flagged for review because {', '.join(reasons[:4])}. "
        f"The risk score is {case_evidence['risk_score']}, and the recommended action is: "
        f"{case_evidence['recommended_action']}. This does not prove fraud, but supports referral triage."
    )


def generate_explanation(case_evidence: Dict[str, Any]) -> str:
    """Public, stable name for the deterministic generator used by the API."""
    return generate_referral_explanation_stub(case_evidence)

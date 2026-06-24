"""Pydantic schemas for the quote-manipulation triage layer.

Two validation boundaries:
  1. CaseEvidence          - the structured facts handed to the LLM (model input)
  2. StructuredExplanation - the parsed, typed form of the v3 structured output
"""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, field_validator, ValidationError

__all__ = ["CaseEvidence", "StructuredExplanation",
           "parse_structured_explanation", "ValidationError"]


class CaseEvidence(BaseModel):
    customer_id: int
    risk_score: float = Field(ge=0.0, le=1.0)
    recommended_action: str = Field(min_length=1)
    n_quotes: int = Field(ge=1)
    first_to_lowest_premium_drop_pct: float
    largest_single_drop_pct: float
    material_edit_count: int = Field(ge=0)
    suspicious_field_edit_count: int = Field(ge=0)
    suspicious_step_ratio: float = Field(ge=0.0)
    mileage_decrease_count: int = Field(ge=0)
    accident_decrease_count: int = Field(ge=0)
    experience_increase_count: int = Field(ge=0)
    age_band_cross_count: int = Field(ge=0)
    consistency_violation_count: int = Field(ge=0)
    quote_sequence: List[dict] = Field(min_length=1)


class StructuredExplanation(BaseModel):
    """Typed form of the v3_structured analyst output (REASONS / ACTION)."""
    reasons: List[str] = Field(min_length=1, max_length=3)
    action: str = Field(min_length=1)

    @field_validator("reasons")
    @classmethod
    def each_reason_under_20_words(cls, value: List[str]) -> List[str]:
        for r in value:
            if not r.strip():
                raise ValueError("reason must not be empty")
            if len(r.split()) > 20:
                raise ValueError(f"reason exceeds 20-word limit: {r!r}")
        return value


def parse_structured_explanation(text: str) -> StructuredExplanation:
    """Parse the fixed REASONS/ACTION text into a validated object.

    Raises pydantic ValidationError if the text does not satisfy the schema.
    """
    reasons: List[str] = []
    action = ""
    in_reasons = False
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("REASONS:"):
            in_reasons = True
            continue
        if line.upper().startswith("ACTION:"):
            in_reasons = False
            action = line.split(":", 1)[1].strip()
            continue
        if in_reasons and line.startswith("-"):
            reasons.append(line.lstrip("-").strip())
    return StructuredExplanation(reasons=reasons, action=action)
"""FastAPI service for the quote-manipulation triage layer.

Serves detection, explanation, and evaluation over HTTP:

  GET  /health      liveness check
  GET  /model-info  which model backs /score, and how it was selected
  POST /score       a raw quote journey in -> risk score, flag, action, and CaseEvidence
  POST /explain     CaseEvidence in -> grounded explanation + self-evaluation scores
  POST /evaluate    explanation + CaseEvidence in -> the four rubric scores + gate verdict

The same pydantic contracts, feature code, generator, and scorers the notebook and
CI use are reused here, so the API cannot drift from the trained/evaluated behaviour.
/score runs the trained model; /explain uses the deterministic generator (no API key).

The two combine into the full pipeline: POST a journey to /score, take the
`case_evidence` it returns, and POST that to /explain for an analyst-ready write-up.

Run locally:
    PYTHONPATH=. uvicorn src.qm.api.main:app --reload
Interactive docs:
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.qm.schema import CaseEvidence
from src.qm.generation import generate_explanation
from src.qm.features import journey_to_features
from src.qm.scoring import score_feature_vector, MODEL_INFO
from src.qm.evaluation import (
    score_groundedness,
    score_factual_correctness,
    score_conciseness,
    score_format_compliance,
)

# Promotion gate thresholds, mirrored from promoted_prompt.json so the API's
# pass/fail verdict matches the one used to promote a prompt.
GATE_THRESHOLDS = {
    "groundedness": 1.0,
    "factual_correctness": 0.9,
    "conciseness": 0.85,
}

app = FastAPI(
    title="Quote Manipulation Triage API",
    description=(
        "Explains and quality-scores flagged motor-insurance quote journeys. "
        "Deterministic and key-free: reuses the project's pydantic contracts, "
        "generator, and evaluation rubric."
    ),
    version="1.2.0",
)

# The recruiter-facing demo page, served at the site root. /docs stays for engineers.
_INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    """Plain-English demo UI: pick a journey, see the model's verdict and why."""
    try:
        return _INDEX_HTML.read_text(encoding="utf-8")
    except FileNotFoundError:
        return HTMLResponse("<h1>Quote Manipulation Triage API</h1><p>See <a href='/docs'>/docs</a>.</p>")


# ------------------------------ score models -------------------------------

class QuoteStep(BaseModel):
    """One quote attempt in a journey: the rating factors the customer entered,
    the premium the pricing engine returned, and when it happened."""
    driver_age: int = Field(ge=16, le=100)
    driver_experience: int = Field(ge=0, le=80)
    previous_accidents: int = Field(ge=0)
    annual_mileage: float = Field(ge=0, description="Annual mileage in x1000 km")
    car_age: int = Field(ge=0)
    premium: float = Field(gt=0, description="Quoted premium returned for this attempt")
    quote_time: datetime = Field(description="ISO 8601 timestamp of the quote, e.g. 2026-03-01T09:00:00")


class ScoreRequest(BaseModel):
    quotes: List[QuoteStep] = Field(min_length=1, description="The quote journey, in order")
    customer_id: int = Field(default=0, description="Optional identifier, echoed back")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": 662,
                "quotes": [
                    {"driver_age": 24, "driver_experience": 2, "previous_accidents": 2,
                     "annual_mileage": 18.0, "car_age": 5, "premium": 1300.0,
                     "quote_time": "2026-03-01T09:00:00"},
                    {"driver_age": 24, "driver_experience": 3, "previous_accidents": 1,
                     "annual_mileage": 14.0, "car_age": 5, "premium": 1100.0,
                     "quote_time": "2026-03-01T09:12:00"},
                    {"driver_age": 25, "driver_experience": 4, "previous_accidents": 0,
                     "annual_mileage": 8.0, "car_age": 5, "premium": 720.0,
                     "quote_time": "2026-03-01T09:21:00"},
                ],
            }
        }
    }


class ScoreResponse(BaseModel):
    customer_id: int
    risk_score: float = Field(description="Model probability the journey is suspicious (0-1)")
    flag: bool = Field(description="True if risk_score clears the flag threshold")
    recommended_action: str
    model_name: str
    case_evidence: CaseEvidence = Field(description="Ready to POST straight to /explain")


# ----------------------------- response models -----------------------------

class ExplanationScores(BaseModel):
    groundedness: int = Field(description="1 if no accusatory language, else 0")
    factual_correctness: int = Field(description="1 if every stated number appears in the evidence, else 0")
    conciseness: int = Field(description="1 if within the word limit, else 0")


class ExplainResponse(BaseModel):
    customer_id: int
    explanation: str
    llm_mode: str = Field(description="Always 'stub' for this service (deterministic, key-free)")
    scores: ExplanationScores


class EvaluateRequest(BaseModel):
    explanation: str = Field(min_length=1, description="The explanation text to score")
    case_evidence: CaseEvidence = Field(description="The evidence the explanation must stay grounded in")
    prompt_version: Optional[str] = Field(
        default=None,
        description="If 'v3_structured', the REASONS/ACTION format check also runs",
    )


class EvaluateResponse(BaseModel):
    groundedness: int
    factual_correctness: int
    conciseness: int
    format_compliance: Optional[int] = Field(
        description="1/0 for v3_structured, null for other prompt versions"
    )
    passed_gate: bool = Field(description="True if the explanation clears the promotion thresholds")


# -------------------------------- endpoints --------------------------------

@app.get("/health")
def health():
    """Liveness check for monitoring and deploy health probes."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Which model backs /score, how it was selected, and its flag threshold."""
    return MODEL_INFO


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Score a raw quote journey end to end.

    Reconstructs the per-step edits from the submitted quotes, builds the same 31
    journey features the model was trained on, runs the model, and returns the risk
    score, flag, recommended action, and a CaseEvidence payload that can be sent
    straight to /explain. The request is validated at the boundary, so a malformed
    journey is rejected with a 422 rather than scored on bad input.
    """
    quotes = [q.model_dump() for q in req.quotes]
    built = journey_to_features(quotes, customer_id=req.customer_id)
    scored = score_feature_vector(built["feature_vector"])

    evidence = CaseEvidence(
        customer_id=req.customer_id,
        risk_score=scored["risk_score"],
        recommended_action=scored["recommended_action"],
        quote_sequence=built["quote_sequence"],
        **built["evidence_fields"],
    )
    return ScoreResponse(
        customer_id=req.customer_id,
        risk_score=scored["risk_score"],
        flag=scored["flag"],
        recommended_action=scored["recommended_action"],
        model_name=MODEL_INFO["model_name"],
        case_evidence=evidence,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(evidence: CaseEvidence):
    """Generate a grounded referral explanation for one flagged case.

    The request body is validated against CaseEvidence before anything runs, so a
    malformed case is rejected with a 422 rather than producing a bad explanation.
    The generated text is immediately scored against the same rubric used in
    evaluation, so each response carries its own quality signal.
    """
    evidence_dict = evidence.model_dump()
    text = generate_explanation(evidence_dict)
    scores = ExplanationScores(
        groundedness=score_groundedness(text),
        factual_correctness=score_factual_correctness(text, evidence_dict),
        conciseness=score_conciseness(text),
    )
    return ExplainResponse(
        customer_id=evidence.customer_id,
        explanation=text,
        llm_mode="stub",
        scores=scores,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    """Score any explanation against the evidence it claims to be grounded in.

    Works on an externally supplied explanation too, not only ones this service
    generated, so it can act as a standalone quality gate.
    """
    evidence_dict = req.case_evidence.model_dump()
    groundedness = score_groundedness(req.explanation)
    factual = score_factual_correctness(req.explanation, evidence_dict)
    concise = score_conciseness(req.explanation)
    fmt = score_format_compliance(req.explanation, req.prompt_version) if req.prompt_version else None

    passed = (
        groundedness >= GATE_THRESHOLDS["groundedness"]
        and factual >= GATE_THRESHOLDS["factual_correctness"]
        and concise >= GATE_THRESHOLDS["conciseness"]
    )
    return EvaluateResponse(
        groundedness=groundedness,
        factual_correctness=factual,
        conciseness=concise,
        format_compliance=fmt,
        passed_gate=passed,
    )

"""FastAPI service for the quote-manipulation triage layer.

Serves the V3 triage layer over HTTP, key-free and deterministic:

  GET  /health    liveness check
  POST /explain   CaseEvidence in -> grounded explanation + self-evaluation scores
  POST /evaluate  explanation + CaseEvidence in -> the four rubric scores + gate verdict

The same pydantic contracts, generator, and scorers the notebook and CI use are
reused here, so the API cannot drift from the evaluated behaviour. No external
model is called; the deterministic generator backs /explain.

Run locally:
    PYTHONPATH=. uvicorn src.qm.api.main:app --reload
Interactive docs:
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.qm.schema import CaseEvidence
from src.qm.generation import generate_explanation
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
    version="1.0.0",
)


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

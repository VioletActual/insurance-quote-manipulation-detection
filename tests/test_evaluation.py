from src.qm.evaluation import (
    score_groundedness,
    score_factual_correctness,
    score_conciseness,
    score_format_compliance,
)


# --- groundedness ---

def test_banned_phrase_fraud_confirmed_fails():
    assert score_groundedness("fraud confirmed") == 0

def test_banned_phrase_deliberate_fails():
    assert score_groundedness("This was a deliberate attempt.") == 0

def test_banned_phrase_manipulat_fails():
    assert score_groundedness("The customer was manipulating the system.") == 0

def test_banned_phrase_intentional_fails():
    assert score_groundedness("This appears intentional.") == 0

def test_banned_phrase_taking_advantage_fails():
    assert score_groundedness("The customer was taking advantage of the system.") == 0

def test_clean_explanation_passes_groundedness():
    assert score_groundedness("This journey may warrant review based on behavioural signals.") == 1

def test_case_insensitive_banned_phrase():
    assert score_groundedness("DELIBERATE attempt detected.") == 0


# --- factual correctness ---

def test_number_in_evidence_passes():
    evidence = {"n_quotes": 5, "risk_score": 0.82}
    assert score_factual_correctness("The customer made 5 quote attempts.", evidence) == 1

def test_hallucinated_number_fails():
    evidence = {"n_quotes": 5, "risk_score": 0.82}
    assert score_factual_correctness("The customer made 9 quote attempts.", evidence) == 0

def test_no_numbers_in_text_passes():
    evidence = {"n_quotes": 5}
    assert score_factual_correctness("This journey warrants review.", evidence) == 1

def test_nested_evidence_number_passes():
    evidence = {"quote_sequence": [{"premium": 450.0}]}
    assert score_factual_correctness("The premium was 450.0.", evidence) == 1


# --- conciseness ---

def test_short_text_passes_conciseness():
    assert score_conciseness("Short explanation here.") == 1

def test_long_text_fails_conciseness():
    assert score_conciseness("word " * 100) == 0

def test_exactly_90_words_passes():
    assert score_conciseness("word " * 90) == 1

def test_91_words_fails():
    assert score_conciseness("word " * 91) == 0


# --- format compliance ---

def test_v3_correct_format_passes():
    text = "REASONS:\n- Reason one here.\n- Reason two here.\nACTION: Escalate for manual review."
    assert score_format_compliance(text, "v3_structured") == 1

def test_v3_missing_reasons_header_fails():
    text = "- Reason one.\nACTION: Escalate."
    assert score_format_compliance(text, "v3_structured") == 0

def test_v3_missing_action_fails():
    text = "REASONS:\n- Reason one.\n- Reason two."
    assert score_format_compliance(text, "v3_structured") == 0

def test_non_v3_returns_none():
    assert score_format_compliance("anything", "v1_basic") is None
    assert score_format_compliance("anything", "v2_grounded_concise") is None
import re

BANNED_PHRASES = [
    "fraud confirmed", "fraudulent", "intentional fraud", "deliberately lied",
    "committed fraud", "is lying", "dishonest customer",
    "deliberate", "intentional", "manipulat", "taking advantage", "trying to manipulate"
]


def extract_numbers(text):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


def evidence_number_set(case_evidence):
    nums = set()

    def collect(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            nums.add(round(float(value), 2))
            nums.add(round(float(value), 0))
        elif isinstance(value, dict):
            for v in value.values():
                collect(v)
        elif isinstance(value, list):
            for v in value:
                collect(v)

    collect(case_evidence)
    return nums


def score_groundedness(text):
    return 0 if any(term in text.lower() for term in BANNED_PHRASES) else 1


def score_factual_correctness(text, case_evidence):
    claimed = extract_numbers(text)
    if not claimed:
        return 1
    valid = evidence_number_set(case_evidence)
    for n in claimed:
        if round(n, 2) not in valid and round(n, 0) not in valid:
            return 0
    return 1


def score_conciseness(text, max_words=90):
    return 1 if len(text.split()) <= max_words else 0


def score_format_compliance(text, prompt_version):
    if prompt_version != "v3_structured":
        return None
    has_reasons = text.strip().startswith("REASONS:")
    has_action = "ACTION:" in text
    bullet_count = text.count("\n-") + text.count("\n -")
    return 1 if (has_reasons and has_action and 1 <= bullet_count <= 3) else 0
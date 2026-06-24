# Behavioural Detection of Suspicious Quote Journeys in Motor Insurance

![CI](https://github.com/VioletActual/insurance-quote-manipulation-detection/actions/workflows/ci.yml/badge.svg)

**In one line:** an end-to-end system that detects suspicious quote-manipulation behaviour in motor insurance, then uses a language model to write analyst-ready explanations for each flagged case, with automated quality gates, experiment tracking, and CI.

It brings together three things usually shown separately: applied machine learning on a realistic imbalanced detection problem, an evaluation pipeline for language-model outputs, and an automated testing setup that runs on every code change.

If you only read one section, read **V3** below. That is the newest work.

---

## Skills this project demonstrates

| Skill area | Where to see it |
|------------|-----------------|
| Applied ML | imbalanced classification, model comparison, PR-AUC and F1 |
| Feature engineering | journey-level behavioural features from raw quotes |
| Fraud / risk thinking | premium-lowering detection, analyst triage framing |
| Evaluation | drift test, human-vs-automatic scoring of LLM outputs |
| LLM product sense | versioned prompts, grounded explanations, quality gates |
| MLOps (light) | MLflow tracking, prompt promotion, reproducibility |
| Software engineering | `src/` modules, tests, CI, deterministic fallback |
| Typed data contracts | pydantic schemas validating the evidence fed to the model and the structured output it returns |
| Communication | plain-English problem framing and honest limitations |

---

## What's in this repo

| Part | What it does | Headline result |
|------|--------------|-----------------|
| **V3** (`QuoteManipulationV3.ipynb`) | Language-model triage layer that explains flagged cases, with evaluation and tracking | Winning prompt selected by an automated quality gate |
| **V2** (`QuoteManipulationV2.ipynb`) | The detection model that finds suspicious quote journeys | Random Forest, F1 0.709, PR-AUC 0.811 |
| **CI** (`.github/workflows/ci.yml`) | Runs the test suite automatically on every push | 30 tests passing |

Repository layout:

    src/qm/      reusable, tested code (scoring functions, prompt library, schemas)
    tests/       automated tests for that code
    .github/     the GitHub Actions setup that runs the tests
    *.ipynb      the full notebooks (V2 detection, V3 triage)

---

## The problem in plain terms

Car insurance prices depend on things like the driver's age, years of experience, past accidents, and annual mileage.

Someone getting a quote online can run it several times, changing those answers slightly each time, until the price drops. If that is done to game the system rather than to correct a genuine mistake, the insurer ends up selling underpriced policies. That is a real pricing-integrity and fraud risk.

Most systems check each quote on its own. This project instead looks at the **whole journey**: the full series of quote attempts from one person, and how their answers change along the way. That is where the suspicious pattern actually shows up.

Typical warning signs: mileage quietly dropping, past accidents disappearing, experience going up, age nudged just over a pricing threshold (24 to 25, 39 to 40), or several important answers changed in a short space of time.

---

## V3: the language-model triage layer (newest work)

The detection model in V2 can flag a case. But a flag on its own is not useful to the person who has to review it. V3 answers the next question: **why was this case flagged, written in a way an analyst can act on?**

How it works, and why each step matters:

**1. Turn a flagged case into evidence.**
Each flagged journey is converted into a clean set of facts (number of quotes, premium drop, what was edited, and so on). The language model only ever sees these facts, never the answer, so it cannot make things up beyond the evidence. These facts are validated against a typed pydantic schema before they reach the model, so a malformed case is caught early rather than quietly producing a bad explanation.

**2. Ask the model to explain, using versioned prompts.**
The instructions given to the model are saved as numbered versions with notes on what changed, the same way code is versioned. Three versions were compared, from a loose first attempt to a strict one that forces a fixed "reasons and recommended action" format.

**3. Score every explanation automatically.**
Each explanation is graded on four things: it must not accuse anyone of fraud, every number it states must come from the evidence, it must be short, and (for the strict version) it must follow the required format.

**4. Check the automatic scoring against a human.**
60 explanations were hand-labelled and compared with the automatic scores. This caught a real weakness: the automatic "no accusations" check originally agreed with the human only **68%** of the time, because it was missing words like "deliberate" and "manipulating". After fixing that, agreement reached **100%**. The "facts must match evidence" check agrees **82%** of the time; the remaining gap is the model misreading numbers (for example calling a high-risk score "low risk"), which a simple word check cannot catch. This is exactly the kind of honest limitation a real evaluation surfaces.

**5. Only promote a prompt that passes the bar.**
A prompt version is allowed through only if it scores perfectly on no-accusations and clears high marks on the other checks (90%+ factual correctness, 85%+ conciseness). Concretely, "promoted" means the winning version is written to a `promoted_prompt.json` file, the single pointer a downstream service or the notebook would read to know which prompt to use. Nothing is chosen by gut feel; it has to clear the gate.

**6. Track everything.**
Every version and its scores are logged with MLflow, including the exact prompt text, so any result can be reproduced and compared later.

**The end product** is `referral_queue.csv`: the flagged cases ranked by risk, each with a clear written explanation attached. That is the list a pricing-integrity team would actually work through.

### Example output

A single (anonymised) row from the referral queue:

| Risk rank | Key changes | Premium drop | Explanation (excerpt) |
|-----------|-------------|--------------|------------------------|
| 1 | 6 quotes; mileage, accidents, experience all reduced across journey | 31.6% | "REASONS: The customer had 7 suspicious field edits. The customer had a high suspicious step ratio of 1.0. The largest single drop in premium was 16.4%. ACTION: Escalate for manual review." |

In short: V3 is a small but complete example of building, testing, and quality-gating a language-model feature, rather than just calling a model and hoping the output is good.

---

## V2: the detection model underneath

**The approach:** simulate realistic quote journeys, turn each journey into behavioural features, screen them with simple rules first, then train and compare machine-learning models, then test how well it holds up when behaviour changes.

**Why the data is synthetic and still hard.** Real insurer quote streams are not public, so journeys are simulated. To stop the model cheating, a hidden behavioural "intent" signal is built in, so the suspicious label cannot be read straight off the visible features, and normal and suspicious journeys are deliberately made to overlap. About 4.5% of journeys are labelled suspicious, which creates a realistic class-imbalance setting for a fraud-style detection problem.

**What the model looks at:** how many quotes were run, how far the premium dropped, whether mileage fell, whether accidents were removed, whether experience rose, whether age crossed a pricing threshold, how quickly the quotes were run, and more.

**Model choice.** Logistic Regression and Histogram Gradient Boosting were both tried as comparisons. Random Forest was selected because it handled the nonlinear behavioural interactions best while staying interpretable through feature importance and case-level evidence extraction, which matters for a triage use case where a human needs to understand the flag.

**Results:**

| Approach | Precision | Recall | F1 | PR-AUC |
|----------|-----------|--------|----|--------|
| Simple rules | 0.800 | 0.178 | 0.291 | — |
| **Random Forest** | **0.824** | **0.622** | **0.709** | **0.811** |

The simple rules only catch the most obvious cases (high precision, low recall). The Random Forest picks up the subtler patterns the rules miss. Layering rules and machine learning this way mirrors how insurers actually do it.

**Robustness check:** when customer behaviour is shifted to simulate change over time, performance dips but stays usable (F1 0.655). That honesty about degradation matters more than a single headline score.

**Methodology note:** metrics are reported on a held-out test set, with PR-AUC used as the primary measure because of the class imbalance.

### Error analysis

Beyond headline metrics, the notebook reviews where the model goes wrong, which matters more than the score for a risk model:

| Error type | Pattern | Likely reason |
|------------|---------|---------------|
| False positive | Customer 958: 6 quotes, material edits to mileage and experience, 19.6% premium drop, but labelled normal | Behavioural signals overlapped with genuine shopping; hidden intent signal marked this as non-suspicious despite journey shape |
| False positive | Customer 801: 5 quotes, 7 material edits including accidents and experience, 18.4% drop, but labelled normal | High edit count over-weighted; some edits moved the premium back up, weakening the manipulation signal |
| False negative | Customer 231: only 3 quotes, just 2 material edits, despite a 19.3% premium drop and suspicious step ratio of 1.0 | Short journey with few edits produced a weak feature signal; model scored it 0.493, just below the flag threshold |
| False negative | Customer 392: 6 quotes but only 2 material edits; manipulation concentrated in mileage reduction with a consistency violation at the final quote | Sparse material edits kept the score low despite sustained mileage manipulation; the consistency violation arrived too late in the journey to push the score over the threshold |
---

## CI: automated testing

Every time code is pushed to GitHub, a fresh environment is set up automatically and the test suite runs: 30 tests covering the scoring functions, the prompt library, and the pydantic schemas that validate model input and output. The green badge at the top of this page shows the current status.

The tests are built to run without calling any paid model, using a built-in deterministic fallback. That keeps the automated checks free, fast, and reliable, and means they never depend on an external key.

---

## How to run it

**Notebooks (Google Colab or Jupyter):**

1. Open `QuoteManipulationV2.ipynb` or `QuoteManipulationV3.ipynb`
2. Make sure `car_insurance_premium_dataset.csv` is available
3. Restart the kernel and run all cells

V3 uses a free language model (Llama 3.1 via Groq) for live explanation generation. Set a `GROQ_API_KEY`, or run with the built-in fallback for a key-free version.

**Tests (from the project root):**

    PYTHONPATH=. pytest tests/ -v

---

## Honest limitations and next steps

The quote journeys are simulated, not real insurer data. The pricing formula is simplified, and the data leaves out factors like postcode, occupation, and no-claims discount. This is a realistic prototype that demonstrates the full workflow, not a production system.

A natural next step on the evaluation side would be to classify the language model's factual errors into categories (numeric mismatch, unsupported claim, wrong risk interpretation, missing evidence) rather than a single pass/fail, which would make the automatic checks more informative.

---

## Changelog

**v3.1** — Added pydantic schema validation at both model boundaries: `CaseEvidence` validates the structured facts passed to the model, and `StructuredExplanation` parses and validates the v3 REASONS/ACTION output. The format-compliance check now uses schema validation rather than string matching. Six schema tests added (24 to 30), covered by CI.
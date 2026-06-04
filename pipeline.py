"""
pipeline.py
-----------
End-to-end AI Triage Pipeline:

  Patient Input
      ↓
  LLM Feature Extraction      (llm_extractor.py)
      ↓
  Structured Features (JSON)
      ↓
  Triage Engine (Rules)       (triage_engine.py)
      ↓
  LLM Clinician Summary       (llm_summary.py)
      ↓
  Final Output
"""

import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

from llm_extractor import extract_features, extract_features_mock
from llm_summary import generate_clinician_summary, generate_summary_mock
from triage_engine import rule_based_triage, TriageResult


@dataclass
class TriagePipelineOutput:
    patient_input: str
    extracted_features: dict
    triage_result: dict          # TriageResult as dict
    clinician_summary: str
    latency_ms: float
    mode: str                    # "live" | "mock"


def run_triage_pipeline(
    patient_text: str,
    api_key: Optional[str] = None,
    use_mock: bool = False,
) -> TriagePipelineOutput:
    """
    Run the full AI triage pipeline on a patient's symptom description.

    Parameters
    ----------
    patient_text : str
        Raw patient-reported symptom text.
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    use_mock : bool
        If True, uses keyword-based extractors (no API calls needed).

    Returns
    -------
    TriagePipelineOutput
        Complete pipeline output including features, triage, and summary.

    Example
    -------
    >>> result = run_triage_pipeline(
    ...     "I have severe chest pain and can't breathe",
    ...     use_mock=True
    ... )
    >>> result.triage_result["category"]
    'emergency'
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: LLM Feature Extraction
    # ------------------------------------------------------------------
    if use_mock:
        features = extract_features_mock(patient_text)
    else:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key=. "
                "Use use_mock=True for testing without a key."
            )
        features = extract_features(patient_text, api_key=key)

    # ------------------------------------------------------------------
    # Step 2: Rule-based Triage Engine
    # ------------------------------------------------------------------
    triage_result: TriageResult = rule_based_triage(features)

    # ------------------------------------------------------------------
    # Step 3: Clinician Copilot Summary
    # ------------------------------------------------------------------
    if use_mock:
        summary = generate_summary_mock(features, triage_result)
    else:
        summary = generate_clinician_summary(features, triage_result, api_key=api_key)

    latency_ms = (time.time() - t0) * 1000

    return TriagePipelineOutput(
        patient_input=patient_text,
        extracted_features=features,
        triage_result=asdict(triage_result),
        clinician_summary=summary,
        latency_ms=round(latency_ms, 2),
        mode="mock" if use_mock else "live",
    )


def print_pipeline_output(output: TriagePipelineOutput) -> None:
    """Pretty-print the pipeline output to the terminal."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  AI TRIAGE SYSTEM — PIPELINE OUTPUT")
    print(sep)
    print(f"  Patient Input : {output.patient_input}")
    print(f"  Mode          : {output.mode.upper()}")
    print(f"  Latency       : {output.latency_ms} ms")
    print(sep)

    print("\n[1] EXTRACTED FEATURES")
    for k, v in output.extracted_features.items():
        print(f"    {k:<22}: {v}")

    tr = output.triage_result
    print("\n[2] TRIAGE DECISION")
    print(f"    Category     : {tr['category'].upper()}")
    print(f"    Confidence   : {tr['confidence']}")
    print(f"    Rationale    : {tr['rationale']}")
    print(f"    Action       : {tr['recommended_action']}")
    if tr["red_flags"]:
        print(f"    ⚠ Red Flags  : {', '.join(tr['red_flags'])}")

    print("\n[3] CLINICIAN SUMMARY")
    print(f"    {output.clinician_summary}")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        "My neck hurts badly and I can't move it since morning",
        "Severe chest pain and difficulty breathing for the last 20 minutes",
        "I have a mild headache and runny nose since yesterday",
        "Feeling extremely fatigued with high fever of 104°F for 2 days",
        "Slight nausea after eating, feels better now",
    ]

    for case in test_cases:
        result = run_triage_pipeline(case, use_mock=True)
        print_pipeline_output(result)

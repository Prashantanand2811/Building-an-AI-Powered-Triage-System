"""
ab_test.py
----------
A/B Test: Rule-only baseline vs LLM + Hybrid pipeline.

Baseline (A): Keyword rules only (no LLM feature extraction).
Variant  (B): LLM feature extraction + rule-based triage (hybrid).

Metrics:
  - Triage accuracy
  - Time to decision (latency)
  - Decision consistency
  - Emergency false-negative rate (guardrail metric)
"""

import csv
import time
from collections import defaultdict
from pathlib import Path

from llm_extractor import extract_features_mock
from triage_engine import rule_based_triage, EMERGENCY


# ---------------------------------------------------------------------------
# Baseline: pure keyword rules, no LLM extraction step
# ---------------------------------------------------------------------------
EMERGENCY_KW = {
    "chest pain", "chest tightness", "can't breathe", "difficulty breathing",
    "shortness of breath", "coughing blood", "vomiting blood", "unconscious",
    "seizure", "overdose", "anaphylaxis", "facial drooping", "slurred speech",
}
URGENT_KW = {
    "severe", "high fever", "neck pain", "can't move", "abdominal pain",
    "back pain", "urinary pain", "rash spreading", "infected wound",
    "dizziness", "fainting", "palpitations",
}
VIDEO_KW = {
    "sore throat", "cough", "nausea", "headache", "muscle ache",
    "fatigue", "runny nose", "diarrhea", "burning when urinating",
    "mild fever", "mild pain",
}


def rule_only_triage(text: str) -> str:
    """Baseline: simple keyword scan, no LLM feature extraction."""
    t = text.lower()
    for kw in EMERGENCY_KW:
        if kw in t:
            return "emergency"
    for kw in URGENT_KW:
        if kw in t:
            return "urgent_doctor_review"
    for kw in VIDEO_KW:
        if kw in t:
            return "video_consult"
    return "self_care"


def hybrid_triage(text: str, severity_override: str = None) -> str:
    """Variant B: LLM mock extraction + rule engine."""
    features = extract_features_mock(text)
    if severity_override:
        features["severity"] = severity_override
    result = rule_based_triage(features)
    return result.category


# ---------------------------------------------------------------------------
# A/B test runner
# ---------------------------------------------------------------------------
def run_ab_test(records: list[dict]) -> dict:
    results_a, results_b = [], []
    times_a, times_b = [], []

    for rec in records:
        text = rec["patient_text"]
        severity = rec.get("severity")
        true_cat = rec["true_category"]

        t0 = time.perf_counter()
        pred_a = rule_only_triage(text)
        times_a.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pred_b = hybrid_triage(text, severity_override=severity)
        times_b.append(time.perf_counter() - t0)

        results_a.append((true_cat, pred_a))
        results_b.append((true_cat, pred_b))

    def accuracy(results):
        return sum(1 for t, p in results if t == p) / len(results)

    def emergency_fn_rate(results):
        em_true = [(t, p) for t, p in results if t == EMERGENCY]
        if not em_true:
            return 0.0
        fn = sum(1 for t, p in em_true if p != EMERGENCY)
        return fn / len(em_true)

    def consistency(results_x, results_y):
        agree = sum(1 for (_, px), (_, py) in zip(results_x, results_y) if px == py)
        return agree / len(results_x)

    return {
        "n": len(records),
        "baseline": {
            "accuracy": accuracy(results_a),
            "avg_latency_ms": sum(times_a) / len(times_a) * 1000,
            "emergency_fn_rate": emergency_fn_rate(results_a),
        },
        "hybrid": {
            "accuracy": accuracy(results_b),
            "avg_latency_ms": sum(times_b) / len(times_b) * 1000,
            "emergency_fn_rate": emergency_fn_rate(results_b),
        },
        "decision_consistency": consistency(results_a, results_b),
    }


def print_ab_report(ab: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  A/B TEST RESULTS: Baseline vs Hybrid Pipeline")
    print(sep)
    print(f"  Total cases: {ab['n']}")
    print()

    headers = ["Metric", "Baseline (Rules Only)", "Hybrid (LLM + Rules)", "Δ"]
    print(f"  {headers[0]:<30} {headers[1]:>22} {headers[2]:>22} {headers[3]:>8}")
    print("  " + "-" * 86)

    b, h = ab["baseline"], ab["hybrid"]

    metrics = [
        ("Accuracy", b["accuracy"], h["accuracy"], True),
        ("Avg Latency (ms)", b["avg_latency_ms"], h["avg_latency_ms"], False),
        ("Emergency FN Rate", b["emergency_fn_rate"], h["emergency_fn_rate"], False),
    ]
    for name, bv, hv, higher_is_better in metrics:
        delta = hv - bv
        symbol = ("↑" if delta > 0 else "↓") if higher_is_better else ("↓" if delta > 0 else "↑")
        if name == "Avg Latency (ms)":
            print(f"  {name:<30} {bv:>21.3f} {hv:>21.3f} {symbol}")
        else:
            print(f"  {name:<30} {bv:>21.3%} {hv:>21.3%} {symbol}")

    print(f"\n  Decision Consistency (A vs B): {ab['decision_consistency']:.2%}")
    print()
    print("  Guardrail check:")
    if h["emergency_fn_rate"] < b["emergency_fn_rate"]:
        print("  ✓ Hybrid reduces emergency false negatives vs baseline.")
    else:
        print("  ⚠ Hybrid emergency FN rate not improved — review extraction rules.")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    dataset_path = "data/synthetic_triage_dataset.csv"

    if not Path(dataset_path).exists():
        print("Dataset not found. Generating...")
        from synthetic_dataset import generate_dataset, save_dataset
        recs = generate_dataset(1000)
        save_dataset(recs, dataset_path)

    with open(dataset_path, newline="") as f:
        records = list(csv.DictReader(f))

    ab_results = run_ab_test(records)
    print_ab_report(ab_results)

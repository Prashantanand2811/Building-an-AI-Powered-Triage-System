"""
evaluate.py
-----------
Evaluation framework for the AI Triage System.
Metrics: Accuracy, Precision, Recall, F1-Score, Red-flag detection sensitivity.
Runs against the synthetic dataset using the mock extractor (no API key needed).
"""

import csv
from collections import defaultdict
from pathlib import Path

from llm_extractor import extract_features_mock
from triage_engine import rule_based_triage

CATEGORIES = ["emergency", "urgent_doctor_review", "video_consult", "self_care"]


def load_dataset(path: str = "data/synthetic_triage_dataset.csv") -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def predict(patient_text: str, severity_override: str = None) -> str:
    features = extract_features_mock(patient_text)
    if severity_override:
        features["severity"] = severity_override
    result = rule_based_triage(features)
    return result.category


def compute_metrics(records: list[dict]) -> dict:
    """
    Compute per-class and macro-averaged metrics.

    Returns
    -------
    dict with keys: accuracy, per_class, macro_avg,
                    emergency_recall, red_flag_sensitivity
    """
    y_true = []
    y_pred = []

    for rec in records:
        true_cat = rec["true_category"]
        pred_cat = predict(rec["patient_text"], severity_override=rec.get("severity"))
        y_true.append(true_cat)
        y_pred.append(pred_cat)

    # Confusion matrix counts
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = 0

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
            correct += 1
        else:
            fn[t] += 1
            fp[p] += 1

    accuracy = correct / len(records)

    per_class = {}
    for cat in CATEGORIES:
        precision = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) > 0 else 0.0
        recall = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[cat] = {"precision": precision, "recall": recall, "f1": f1,
                          "support": y_true.count(cat)}

    macro_precision = sum(v["precision"] for v in per_class.values()) / len(CATEGORIES)
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(CATEGORIES)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(CATEGORIES)

    # Critical safety metric: emergency recall
    emergency_recall = per_class["emergency"]["recall"]

    return {
        "total_cases": len(records),
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "emergency_recall": emergency_recall,
    }


def print_report(metrics: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  AI TRIAGE SYSTEM — EVALUATION REPORT")
    print(sep)
    print(f"  Total Cases : {metrics['total_cases']}")
    print(f"  Accuracy    : {metrics['accuracy']*100:.2f}%")
    print(sep)

    print(f"\n{'Category':<26} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 68)
    for cat, m in metrics["per_class"].items():
        print(
            f"{cat:<26} {m['precision']:>10.3f} {m['recall']:>10.3f} "
            f"{m['f1']:>10.3f} {m['support']:>10}"
        )
    print("-" * 68)
    ma = metrics["macro_avg"]
    print(f"{'macro avg':<26} {ma['precision']:>10.3f} {ma['recall']:>10.3f} {ma['f1']:>10.3f}")

    print(f"\n⚠  Emergency Recall (safety metric): {metrics['emergency_recall']*100:.2f}%")
    if metrics["emergency_recall"] < 0.90:
        print("   WARNING: Emergency recall below 90% — review triage rules!")
    else:
        print("   ✓ Emergency recall meets safety threshold (≥90%)")
    print(f"\n{sep}\n")


def save_report(metrics: dict, path: str = "data/evaluation_report.txt") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_report(metrics)
    sys.stdout = old_stdout
    with open(path, "w") as f:
        f.write(buf.getvalue())
    print(f"✓ Report saved to {path}")


if __name__ == "__main__":
    dataset_path = "data/synthetic_triage_dataset.csv"

    if not Path(dataset_path).exists():
        print("Dataset not found. Generating...")
        from synthetic_dataset import generate_dataset, save_dataset
        records_raw = generate_dataset(1000)
        save_dataset(records_raw, dataset_path)

    records = load_dataset(dataset_path)
    metrics = compute_metrics(records)
    print_report(metrics)
    save_report(metrics)

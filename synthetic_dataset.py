"""
synthetic_dataset.py
--------------------
Generates ~1,000 labelled synthetic triage cases for evaluation.
Since real healthcare datasets are restricted, this mirrors the approach
described in the Medium article.
"""

import csv
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Symptom templates per category
# ---------------------------------------------------------------------------
CASE_TEMPLATES = {
    "emergency": [
        "Severe chest pain and difficulty breathing for the last {n} minutes",
        "Crushing chest tightness radiating to left arm, started {n} minutes ago",
        "Coughing up blood and can't breathe properly",
        "Sudden loss of consciousness and seizure",
        "Severe allergic reaction, throat is swelling",
        "Vomiting blood and extreme abdominal pain",
        "Suspected drug overdose, patient unresponsive",
        "Sudden severe headache described as worst ever with vision loss",
        "Left-sided facial drooping and slurred speech, started {n} minutes ago",
        "Difficulty breathing after bee sting, known allergy",
    ],
    "urgent_doctor_review": [
        "Severe neck pain and can't turn my head since this morning",
        "High fever of {temp}°F for 2 days, body aches all over",
        "Severe abdominal pain in lower right side for {n} hours",
        "Significant back pain radiating down the leg, can barely walk",
        "Eye is very red and painful with sensitivity to light",
        "Ear pain with discharge, feels like pressure inside",
        "Urinary pain and burning with blood in urine",
        "Severe headache on one side with nausea and light sensitivity",
        "Rash spreading rapidly across the torso",
        "Persistent high fever in a child under 2 years old",
        "Deep infected wound with red streaks spreading from it",
        "Severe dizziness and fainting when standing up",
    ],
    "video_consult": [
        "Sore throat and mild fever since yesterday",
        "Persistent cough for {n} days with some congestion",
        "Mild stomach ache and nausea after eating",
        "Headache that won't go away for {n} hours",
        "Muscle aches and fatigue, think it might be the flu",
        "Runny nose and sneezing, mild cold symptoms",
        "Mild lower back pain after long day of sitting",
        "Stomach cramps and diarrhea for {n} hours",
        "Itchy rash on arm, not spreading",
        "Mild dizziness when standing up quickly",
        "Burning when urinating, no fever",
        "Mild palpitations lasting a few seconds",
    ],
    "self_care": [
        "Minor scrape on my knee, cleaned it already",
        "Slight headache, probably from not drinking enough water",
        "Feeling a bit tired today, slept poorly last night",
        "Mildly bloated after a large meal",
        "Occasional mild hiccups for the past hour",
        "Small paper cut on my finger",
        "Dry skin on my hands, nothing serious",
        "Slightly sore muscles after working out yesterday",
        "Mild indigestion after eating spicy food",
        "Occasional mild sneezing, probably dust allergy",
    ],
}

SEVERITIES = {
    "emergency": "severe",
    "urgent_doctor_review": ["severe", "moderate"],
    "video_consult": ["moderate", "mild"],
    "self_care": "mild",
}

AGES = list(range(18, 80))


def _fill_template(template: str) -> str:
    return template.format(
        n=random.choice([5, 10, 15, 20, 30, 45, 60, 2, 3, 4]),
        temp=random.choice([101, 102, 103, 104]),
    )


def generate_dataset(n_cases: int = 1000, seed: int = 42) -> list[dict]:
    """
    Generate n_cases synthetic triage records.

    Returns
    -------
    list of dicts with keys:
        patient_text, true_category, severity, age
    """
    random.seed(seed)
    categories = list(CASE_TEMPLATES.keys())

    # Approximate real-world distribution
    weights = {
        "emergency": 0.10,
        "urgent_doctor_review": 0.30,
        "video_consult": 0.40,
        "self_care": 0.20,
    }

    records = []
    for _ in range(n_cases):
        cat = random.choices(categories, weights=list(weights.values()))[0]
        template = random.choice(CASE_TEMPLATES[cat])
        text = _fill_template(template)

        sev = SEVERITIES[cat]
        severity = sev if isinstance(sev, str) else random.choice(sev)

        records.append({
            "patient_text": text,
            "true_category": cat,
            "severity": severity,
            "age": random.choice(AGES),
        })

    random.shuffle(records)
    return records


def save_dataset(records: list[dict], path: str = "data/synthetic_triage_dataset.csv") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_text", "true_category", "severity", "age"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"✓ Saved {len(records)} synthetic cases to {path}")


if __name__ == "__main__":
    records = generate_dataset(1000)
    save_dataset(records)

    # Quick distribution check
    from collections import Counter
    dist = Counter(r["true_category"] for r in records)
    print("\nCategory distribution:")
    for cat, count in sorted(dist.items()):
        print(f"  {cat:<25} {count:>4} ({count/len(records)*100:.1f}%)")

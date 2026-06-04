"""
llm_extractor.py
----------------
Step 1: Transform raw patient text → structured JSON features.
Uses GPT-4o-mini as a feature engineering layer for unstructured health data.
"""

import json
import os
from typing import Optional

import openai

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """You are a clinical NLP assistant. Your job is to extract structured
medical features from a patient's free-text symptom description.

Return ONLY valid JSON (no markdown, no explanation). The JSON must follow this schema exactly:

{
  "symptoms": ["<list of symptoms mentioned>"],
  "severity": "<one of: mild | moderate | severe>",
  "body_part": "<primary body part or region affected>",
  "duration": "<how long the symptoms have persisted, as described>",
  "red_flags": ["<list any life-threatening signals like chest pain, difficulty breathing, etc. Empty list if none>"],
  "age": <integer if mentioned, else null>,
  "existing_conditions": ["<any chronic conditions or medications mentioned>"]
}

Be conservative: only flag red_flags if there is a clear life-threatening indicator.
Do not add information not present in the input.
"""


def extract_features(patient_text: str, api_key: Optional[str] = None) -> dict:
    """
    Extract structured clinical features from a patient's symptom description.

    Parameters
    ----------
    patient_text : str
        Raw text input from the patient (e.g. "My neck hurts badly since morning").
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY environment variable.

    Returns
    -------
    dict
        Structured features matching the JSON schema above.

    Example
    -------
    >>> features = extract_features("Severe chest tightness and left arm pain for 20 minutes")
    >>> features["red_flags"]
    ['chest tightness', 'possible cardiac event']
    """
    client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Patient input: {patient_text}"},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    features = json.loads(raw)
    return features


def extract_features_mock(patient_text: str) -> dict:
    """
    Deterministic mock extractor for testing without an OpenAI key.
    Uses simple keyword matching to produce structured features.
    """
    text_lower = patient_text.lower()

    # Severity detection
    if any(w in text_lower for w in ["severe", "worst", "unbearable", "extremely", "can't move"]):
        severity = "severe"
    elif any(w in text_lower for w in ["moderate", "significant", "bad", "really"]):
        severity = "moderate"
    else:
        severity = "mild"

    # Red flag detection
    red_flags = []
    red_flag_keywords = [
        "chest pain", "chest tightness", "difficulty breathing",
        "shortness of breath", "can't breathe", "vomiting blood",
        "coughing blood", "unconscious", "seizure", "overdose",
        "anaphylaxis", "allergic reaction", "facial drooping",
        "slurred speech", "sudden vision loss", "loss of consciousness",
        "crushing", "radiating to left arm", "unresponsive",
        "throat is swelling", "suspected drug",
    ]
    for kw in red_flag_keywords:
        if kw in text_lower:
            red_flags.append(kw)

    # Simple symptom extraction
    symptom_keywords = {
        "neck pain": "neck",
        "headache": "head",
        "chest pain": "chest",
        "fever": "body",
        "nausea": "stomach",
        "vomiting": "stomach",
        "cough": "chest",
        "back pain": "back",
        "stomach pain": "stomach",
        "abdominal pain": "abdomen",
        "dizziness": "head",
        "fatigue": "general",
        "sore throat": "throat",
        "runny nose": "nose",
        "rash": "skin",
    }
    symptoms = []
    body_part = "general"
    for kw, part in symptom_keywords.items():
        if kw in text_lower:
            symptoms.append(kw)
            body_part = part

    if not symptoms:
        symptoms = [patient_text[:50]]

    # Duration extraction (basic)
    duration = "unknown"
    for phrase in ["since morning", "since yesterday", "for 2 days", "for a week",
                   "for hours", "for days", "for weeks"]:
        if phrase in text_lower:
            duration = phrase
            break

    return {
        "symptoms": symptoms,
        "severity": severity,
        "body_part": body_part,
        "duration": duration,
        "red_flags": red_flags,
        "age": None,
        "existing_conditions": [],
    }

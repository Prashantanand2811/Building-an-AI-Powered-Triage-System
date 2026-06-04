"""
triage_engine.py
----------------
Rule-based triage decision layer.
Deterministic, explainable, safe — by design.
"""

from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Triage categories (ordered by severity)
# ---------------------------------------------------------------------------
EMERGENCY = "emergency"
URGENT_DOCTOR_REVIEW = "urgent_doctor_review"
VIDEO_CONSULT = "video_consult"
SELF_CARE = "self_care"

TRIAGE_PRIORITY = {
    EMERGENCY: 4,
    URGENT_DOCTOR_REVIEW: 3,
    VIDEO_CONSULT: 2,
    SELF_CARE: 1,
}

# ---------------------------------------------------------------------------
# Red-flag symptom keywords
# ---------------------------------------------------------------------------
EMERGENCY_KEYWORDS = {
    "chest pain", "chest tightness", "heart attack", "stroke",
    "difficulty breathing", "shortness of breath", "can't breathe",
    "unconscious", "unresponsive", "severe bleeding", "coughing blood",
    "vomiting blood", "seizure", "paralysis", "sudden confusion",
    "sudden vision loss", "anaphylaxis", "allergic reaction",
    "overdose", "suicidal", "self harm",
}

URGENT_KEYWORDS = {
    "high fever", "fever above 103", "severe pain",
    "neck pain", "neck stiffness", "restricted movement",
    "abdominal pain", "severe headache", "head injury",
    "dizziness", "fainting", "rapid heartbeat", "palpitations",
    "rash", "hives", "swelling", "urinary infection",
    "back pain", "kidney pain", "ear pain", "eye pain",
    "infected wound", "spreading redness",
}

VIDEO_CONSULT_KEYWORDS = {
    "pain", "ache", "discomfort", "nausea", "vomiting",
    "diarrhea", "constipation", "fatigue", "tiredness",
    "cough", "cold", "runny nose", "sore throat",
    "mild fever", "headache", "muscle ache",
}


@dataclass
class TriageResult:
    category: str
    confidence: str          # "high" | "medium" | "low"
    rationale: str
    red_flags: List[str]
    recommended_action: str


def _match_keywords(symptoms: List[str], keyword_set: set) -> List[str]:
    """Return which keywords from the set appear in the symptom list."""
    symptoms_lower = {s.lower() for s in symptoms}
    matches = []
    for kw in keyword_set:
        for sym in symptoms_lower:
            if kw in sym or sym in kw:
                matches.append(kw)
                break
    return list(set(matches))


def rule_based_triage(features: dict) -> TriageResult:
    """
    Apply deterministic rules to structured features extracted by the LLM.

    Parameters
    ----------
    features : dict
        Expected keys: symptoms (list[str]), severity (str),
        body_part (str), duration (str), red_flags (list[str])

    Returns
    -------
    TriageResult
    """
    symptoms: List[str] = features.get("symptoms", [])
    severity: str = features.get("severity", "mild").lower()
    red_flags: List[str] = features.get("red_flags", [])
    age: Optional[int] = features.get("age")

    # ------------------------------------------------------------------
    # Rule 1: Explicit red flags from LLM → emergency
    # ------------------------------------------------------------------
    if red_flags:
        return TriageResult(
            category=EMERGENCY,
            confidence="high",
            rationale=f"Red flags detected: {', '.join(red_flags)}",
            red_flags=red_flags,
            recommended_action="Call emergency services (911) immediately.",
        )

    # ------------------------------------------------------------------
    # Rule 2: Emergency keyword match
    # ------------------------------------------------------------------
    emergency_matches = _match_keywords(symptoms, EMERGENCY_KEYWORDS)
    if emergency_matches:
        return TriageResult(
            category=EMERGENCY,
            confidence="high",
            rationale=f"Emergency symptoms detected: {', '.join(emergency_matches)}",
            red_flags=emergency_matches,
            recommended_action="Call emergency services (911) immediately.",
        )

    # ------------------------------------------------------------------
    # Rule 3: Severe + urgent keywords → urgent doctor review
    # ------------------------------------------------------------------
    urgent_matches = _match_keywords(symptoms, URGENT_KEYWORDS)
    if severity == "severe" or (severity in ("moderate", "high") and urgent_matches):
        return TriageResult(
            category=URGENT_DOCTOR_REVIEW,
            confidence="high" if severity == "severe" else "medium",
            rationale=f"Severity '{severity}' with symptoms: {', '.join(urgent_matches or symptoms[:3])}",
            red_flags=[],
            recommended_action="See a doctor within 2–4 hours. Use urgent care or telehealth.",
        )

    # ------------------------------------------------------------------
    # Rule 4: Urgent keyword match (moderate severity)
    # ------------------------------------------------------------------
    if urgent_matches:
        return TriageResult(
            category=URGENT_DOCTOR_REVIEW,
            confidence="medium",
            rationale=f"Concerning symptoms identified: {', '.join(urgent_matches)}",
            red_flags=[],
            recommended_action="Schedule same-day or next-day doctor appointment.",
        )

    # ------------------------------------------------------------------
    # Rule 5: Moderate severity → video consult
    # ------------------------------------------------------------------
    if severity == "moderate":
        return TriageResult(
            category=VIDEO_CONSULT,
            confidence="medium",
            rationale="Moderate severity with no urgent red flags.",
            red_flags=[],
            recommended_action="Book a video consultation within 24 hours.",
        )

    # ------------------------------------------------------------------
    # Rule 6: Common symptom keywords → video consult
    # ------------------------------------------------------------------
    video_matches = _match_keywords(symptoms, VIDEO_CONSULT_KEYWORDS)
    if video_matches:
        return TriageResult(
            category=VIDEO_CONSULT,
            confidence="medium",
            rationale=f"Non-urgent symptoms: {', '.join(video_matches[:3])}",
            red_flags=[],
            recommended_action="Book a video consultation or visit walk-in clinic.",
        )

    # ------------------------------------------------------------------
    # Rule 7: Default → self care
    # ------------------------------------------------------------------
    return TriageResult(
        category=SELF_CARE,
        confidence="low",
        rationale="Mild or unclear symptoms. No urgent indicators found.",
        red_flags=[],
        recommended_action="Rest, hydrate, and monitor. Visit a doctor if symptoms worsen.",
    )

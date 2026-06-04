"""
llm_summary.py
--------------
Step 3: Generate a doctor-ready clinician copilot summary.
Reduces clinician review time from ~5 min → <2 min.
"""

import os
from typing import Optional

import openai

from triage_engine import TriageResult

SUMMARY_SYSTEM_PROMPT = """You are a clinical documentation assistant.
Given structured patient features and a triage decision, generate a concise,
professional doctor-ready summary in 2-3 sentences.

Format:
"[Age/demographics if available] patient presents with [key symptoms]. 
[Severity and any red flags]. [Triage recommendation and urgency]."

Be factual, clinical, and brief. Do not add assumptions beyond the provided data.
"""


def generate_clinician_summary(
    features: dict,
    triage_result: TriageResult,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a concise, doctor-ready summary for the clinician.

    Parameters
    ----------
    features : dict
        Structured features from llm_extractor.
    triage_result : TriageResult
        Output from the triage engine.
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env var.

    Returns
    -------
    str
        A 2-3 sentence clinical summary.
    """
    client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    user_prompt = f"""
Patient Features:
- Symptoms: {', '.join(features.get('symptoms', []))}
- Severity: {features.get('severity', 'unknown')}
- Body part: {features.get('body_part', 'unknown')}
- Duration: {features.get('duration', 'unknown')}
- Red flags: {', '.join(features.get('red_flags', [])) or 'None'}
- Age: {features.get('age') or 'Not provided'}
- Existing conditions: {', '.join(features.get('existing_conditions', [])) or 'None reported'}

Triage Decision:
- Category: {triage_result.category}
- Rationale: {triage_result.rationale}
- Recommended action: {triage_result.recommended_action}

Write the clinician summary now:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip()


def generate_summary_mock(features: dict, triage_result: TriageResult) -> str:
    """
    Template-based summary for testing without an API key.
    """
    age_str = f"{features['age']}-year-old patient" if features.get("age") else "Patient"
    symptoms_str = ", ".join(features.get("symptoms", ["unspecified symptoms"]))
    duration_str = features.get("duration", "unknown duration")
    severity_str = features.get("severity", "unspecified severity")
    red_flag_str = (
        f" Red flags identified: {', '.join(features['red_flags'])}."
        if features.get("red_flags")
        else " No neurological or cardiac red flags identified."
    )

    category_map = {
        "emergency": "Immediate emergency intervention required. Activate emergency protocol.",
        "urgent_doctor_review": "Urgent clinician review recommended within 2-4 hours.",
        "video_consult": "Video consultation recommended within 24 hours.",
        "self_care": "Self-care monitoring appropriate; advise patient to return if symptoms worsen.",
    }
    action = category_map.get(triage_result.category, triage_result.recommended_action)

    return (
        f"{age_str} presents with {symptoms_str} persisting {duration_str}. "
        f"Severity assessed as {severity_str}.{red_flag_str} "
        f"{action}"
    )

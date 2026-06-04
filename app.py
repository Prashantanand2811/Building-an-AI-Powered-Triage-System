"""
app.py
------
FastAPI REST API for the AI-Powered Triage System.

Endpoints:
  POST /triage          — Run full pipeline on patient text
  POST /triage/batch    — Run pipeline on multiple inputs
  GET  /health          — Health check
  GET  /docs            — Auto-generated Swagger UI (FastAPI default)

Run:
  uvicorn app:app --reload
"""

import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_extractor import extract_features, extract_features_mock
from llm_summary import generate_clinician_summary, generate_summary_mock
from triage_engine import rule_based_triage

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI-Powered Triage System",
    description=(
        "Compresses the patient → clinical decision workflow from minutes to seconds. "
        "Built by Prashant Anand — based on: "
        "https://medium.com/@prashant.anand206/from-symptoms-to-medication-building-an-ai-powered-triage-system"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class TriageRequest(BaseModel):
    patient_text: str = Field(
        ...,
        description="Raw symptom description from the patient.",
        example="My neck hurts badly and I can't move it since morning",
    )
    use_mock: bool = Field(
        default=False,
        description="Use mock extractor (no OpenAI key required). Good for testing.",
    )


class TriageResponse(BaseModel):
    patient_text: str
    extracted_features: dict
    triage_category: str
    triage_confidence: str
    triage_rationale: str
    recommended_action: str
    red_flags: List[str]
    clinician_summary: str
    latency_ms: float
    mode: str


class BatchTriageRequest(BaseModel):
    inputs: List[TriageRequest]


class HealthResponse(BaseModel):
    status: str
    version: str
    openai_key_configured: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns API health status and configuration."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        openai_key_configured=bool(os.environ.get("OPENAI_API_KEY")),
    )


@app.post("/triage", response_model=TriageResponse, tags=["Triage"])
async def triage_patient(request: TriageRequest):
    """
    Run the full AI triage pipeline on a patient's symptom description.

    Pipeline:
      1. LLM Feature Extraction
      2. Rule-based Triage Decision
      3. Clinician Copilot Summary
    """
    t0 = time.time()

    try:
        if request.use_mock:
            features = extract_features_mock(request.patient_text)
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(
                    status_code=400,
                    detail="OPENAI_API_KEY not configured. Set use_mock=true for testing.",
                )
            features = extract_features(request.patient_text, api_key=api_key)

        triage = rule_based_triage(features)

        if request.use_mock:
            summary = generate_summary_mock(features, triage)
        else:
            summary = generate_clinician_summary(features, triage)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.time() - t0) * 1000, 2)

    return TriageResponse(
        patient_text=request.patient_text,
        extracted_features=features,
        triage_category=triage.category,
        triage_confidence=triage.confidence,
        triage_rationale=triage.rationale,
        recommended_action=triage.recommended_action,
        red_flags=triage.red_flags,
        clinician_summary=summary,
        latency_ms=latency_ms,
        mode="mock" if request.use_mock else "live",
    )


@app.post("/triage/batch", response_model=List[TriageResponse], tags=["Triage"])
async def batch_triage(request: BatchTriageRequest):
    """Run the triage pipeline on multiple patient inputs."""
    if len(request.inputs) > 50:
        raise HTTPException(status_code=400, detail="Maximum batch size is 50.")
    results = []
    for item in request.inputs:
        result = await triage_patient(item)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

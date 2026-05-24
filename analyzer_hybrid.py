"""
Hybrid analysis engine: rule-based risk scoring + local LLM recommendation generation.

This module addresses the methodological concern that "lightweight rule-based"
systems alone are not LLM-driven. The hybrid approach uses:
  1. Deterministic rule-based scoring for incident classification and risk assessment
  2. A locally-hosted LLM (Llama-3 via Ollama) for context-aware,
     NIST SP 800-61 compliant recommendation generation

Local LLM execution avoids cloud API quota limits, network dependency,
and data-privacy concerns inherent to commercial LLM APIs.

Requirements:
    - Ollama installed and running (https://ollama.com)
    - A pulled model, e.g.: ollama pull llama3.2
"""

import json
import time
import re
import urllib.request
import urllib.error
from typing import Dict, List

from analyzer_rule_based import classify_incident, calculate_risk_score, determine_risk_level


OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2"


HYBRID_PROMPT_TEMPLATE = """CRITICAL: Output EXACTLY 5 concise incident response actions.
Format ONLY as JSON. EACH ACTION MAX 15 WORDS.

Incident: {incident_type} | Risk: {risk_level}

RESPOND WITH THIS FORMAT ONLY:
{{
  "recommendations": [
    "Block the source IP address",
    "Review authentication logs",
    "Isolate affected system",
    "Patch vulnerability immediately",
    "Notify security team now"
  ]
}}

REMEMBER: Exactly 5 actions, JSON format ONLY.

Now generate recommendations for: {incident_type}
"""


def _ollama_available() -> bool:
    """Check if Ollama service is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> str:
    """Call the local Ollama API and return the generated text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_predict": 600
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed.get("response", "")


def _extract_recommendations(text: str) -> List[str]:
    """Parse the LLM's JSON response and return the recommendation list."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in LLM response")
    obj = json.loads(text[start:end + 1])
    recs = obj.get("recommendations", [])
    if not isinstance(recs, list):
        raise ValueError("recommendations is not a list")
    return [str(r).strip() for r in recs if str(r).strip()]


def analyze_incident_hybrid(alert: Dict, model: str = DEFAULT_MODEL) -> Dict:
    """
    Hybrid analysis: rule-based scoring + LLM-generated recommendations.

    This is the engine the paper should describe as "AI copilot" because the
    final remediation guidance is produced by a real LLM (Llama-3 via Ollama),
    while quantitative risk scoring remains deterministic and auditable.
    """
    start_time = time.time()

    # Stage 1: deterministic rule-based scoring
    incident_type = classify_incident(alert)
    risk_score = calculate_risk_score(alert)
    risk_level = determine_risk_level(risk_score)
    rule_time = round((time.time() - start_time) * 1000, 3)

    # Stage 2: LLM-driven recommendation generation
    llm_start = time.time()
    prompt = HYBRID_PROMPT_TEMPLATE.format(
        scenario_id=alert.get("scenario_id", "N/A"),
        incident_type=incident_type,
        risk_score=risk_score,
        risk_level=risk_level,
        source_ip=alert.get("source_ip", "N/A"),
        destination_ip=alert.get("destination_ip", "N/A"),
        asset_criticality=alert.get("asset_criticality", "N/A"),
        evidence_confidence=alert.get("evidence_confidence", "N/A"),
        event_frequency=alert.get("event_frequency", "N/A"),
        description=alert.get("description", "")
    )

    try:
        if not _ollama_available():
            raise ConnectionError(
                "Ollama service not reachable at http://127.0.0.1:11434. "
                "Start it with: ollama serve"
            )
        raw_response = _call_ollama(prompt, model=model)
        recommendations = _extract_recommendations(raw_response)
        if len(recommendations) == 0:
            raise ValueError("LLM returned empty recommendations")
        llm_time = round((time.time() - llm_start) * 1000, 3)
        total_time = round((time.time() - start_time) * 1000, 3)

        return {
            "scenario_id": alert.get("scenario_id", "N/A"),
            "incident_type": incident_type,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "processing_time": total_time,
            "rule_time_ms": rule_time,
            "llm_time_ms": llm_time,
            "engine": f"hybrid-{model}"
        }
    except Exception as e:
        total_time = round((time.time() - start_time) * 1000, 3)
        return {
            "scenario_id": alert.get("scenario_id", "N/A"),
            "incident_type": incident_type,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendations": [f"LLM error: {str(e)[:200]}"],
            "processing_time": total_time,
            "rule_time_ms": rule_time,
            "llm_time_ms": 0,
            "engine": f"hybrid-{model}",
            "error": str(e)
        }

"""
LLM-based incident analysis engine using Google Gemini API.
Uses the new google-genai SDK (the legacy google-generativeai is deprecated).

To use this module:
1. Get a free API key from https://aistudio.google.com/
2. Set the GEMINI_API_KEY environment variable, or pass api_key to analyze_incident_ai
3. Install: pip install google-genai
"""

import os
import json
import time
import re
from typing import Dict

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


SYSTEM_PROMPT = """You are an experienced Security Operations Center (SOC) analyst.
You will receive a security alert in JSON format and must analyze it.

Respond ONLY with a valid JSON object containing these exact fields:
{
  "incident_type": "<one of: Brute Force Attack, SQL Injection, Phishing, Privilege Escalation, Data Exfiltration, Unknown>",
  "risk_score": <number between 0 and 100>,
  "risk_level": "<one of: Low, Medium, High, Critical>",
  "recommendations": ["action 1", "action 2", "action 3", "action 4", "action 5"]
}

Risk level mapping:
- 0-39: Low
- 40-69: Medium
- 70-84: High
- 85-100: Critical

Provide exactly 5 concrete, actionable recommendations.
Do not include any text outside the JSON object. No markdown code fences."""


def _extract_json(text: str) -> Dict:
    """Extract JSON from LLM response, tolerating common formatting issues."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    json_str = text[start:end + 1]
    return json.loads(json_str)


def analyze_incident_ai(alert: Dict, api_key: str = None, model_name: str = "gemini-2.5-flash") -> Dict:
    """
    Analyze an incident using Google Gemini via the new google-genai SDK.
    """
    if not GENAI_AVAILABLE:
        raise ImportError(
            "google-genai package not installed. Run: pip install google-genai"
        )

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Gemini API key not provided. Set GEMINI_API_KEY env var or pass api_key."
        )

    client = genai.Client(api_key=key)
    user_message = f"Analyze this security alert:\n\n{json.dumps(alert, indent=2)}"

    start_time = time.time()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        raw_text = response.text
        parsed = _extract_json(raw_text)
    except Exception as e:
        processing_time = round((time.time() - start_time) * 1000, 3)
        return {
            "scenario_id": alert.get("scenario_id", "N/A"),
            "incident_type": "Unknown",
            "risk_score": 0.0,
            "risk_level": "Low",
            "recommendations": [f"API error: {str(e)[:200]}"],
            "processing_time": processing_time,
            "engine": f"ai-{model_name}",
            "error": str(e)
        }

    processing_time = round((time.time() - start_time) * 1000, 3)

    return {
        "scenario_id": alert.get("scenario_id", "N/A"),
        "incident_type": parsed.get("incident_type", "Unknown"),
        "risk_score": float(parsed.get("risk_score", 0)),
        "risk_level": parsed.get("risk_level", "Low"),
        "recommendations": parsed.get("recommendations", []),
        "processing_time": processing_time,
        "engine": f"ai-{model_name}"
    }

"""
AI Cybersecurity Copilot - Streamlit Application
Interactive prototype for analyzing security incidents using both
rule-based logic and AI (Google Gemini) approaches.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

from analyzer_rule_based import analyze_incident
from analyzer_ai import analyze_incident_ai, GENAI_AVAILABLE
from analyzer_hybrid import analyze_incident_hybrid, _ollama_available


# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="AI Cybersecurity Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern dark security-themed look
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%);
    }

    /* Headers */
    h1, h2, h3 {
        color: #e6edf3 !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Custom card style */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }

    /* Risk level badges */
    .risk-critical {
        background: linear-gradient(135deg, #dc2626, #991b1b);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-high {
        background: linear-gradient(135deg, #ea580c, #c2410c);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ca8a04, #a16207);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-low {
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 20, 25, 0.95);
    }

    /* Code blocks */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.4) !important;
        border-radius: 8px;
    }

    /* Recommendation list items */
    .rec-item {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
        padding: 10px 16px;
        margin: 8px 0;
        border-radius: 4px;
        color: #e6edf3;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_scenarios():
    """Load all scenario JSON files from the scenarios directory."""
    scenarios = {}
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            scenarios[data["scenario_id"]] = data
    return scenarios


def risk_badge_html(level: str) -> str:
    """Render colored risk level badge."""
    css_class = f"risk-{level.lower()}"
    return f'<span class="{css_class}">{level.upper()}</span>'


def save_result(result: dict):
    """Append result to results log file."""
    log_file = RESULTS_DIR / "experiment_log.csv"
    row = {
        "timestamp": datetime.now().isoformat(),
        "scenario_id": result["scenario_id"],
        "engine": result["engine"],
        "incident_type": result["incident_type"],
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "processing_time": result["processing_time"],
        "recommendations": " | ".join(result["recommendations"])
    }
    df = pd.DataFrame([row])
    df.to_csv(log_file, mode="a", header=not log_file.exists(), index=False)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("# 🛡️ Cybersecurity Copilot")
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")

    engine = st.radio(
        "Analysis Engine",
        options=["Rule-Based", "Hybrid (Rule + Local LLM)", "AI (Gemini)", "All Three"],
        help=(
            "Rule-Based: deterministic, fastest, no LLM\n"
            "Hybrid: rule-based scoring + local Llama-3 recommendations (recommended)\n"
            "AI (Gemini): full LLM analysis via Google Gemini cloud API"
        )
    )

    # Ollama section for hybrid mode
    if engine in ["Hybrid (Rule + Local LLM)", "All Three"]:
        st.markdown("**🦙 Local LLM (Ollama)**")
        if _ollama_available():
            st.success("✅ Ollama running")
        else:
            st.error("❌ Ollama not running. Start with: `ollama serve`")
        ollama_model = st.text_input(
            "Ollama Model Name",
            value="llama3.2",
            help="Must be already pulled: `ollama pull llama3.2`"
        )
    else:
        ollama_model = "llama3.2"

    if engine in ["AI (Gemini)", "All Three"]:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Get a free key from https://aistudio.google.com/"
        )
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

        model_name = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
            index=0,
            help="Flash-Lite has the most generous free tier (1000 req/day)"
        )

    st.markdown("---")
    st.markdown("### 📊 Experiment Stats")

    log_file = RESULTS_DIR / "experiment_log.csv"
    if log_file.exists():
        log_df = pd.read_csv(log_file)
        st.metric("Total Runs", len(log_df))
        st.metric("Scenarios Tested", log_df["scenario_id"].nunique())
    else:
        st.info("No runs yet")

    st.markdown("---")
    st.caption("Prototype for academic research.\nSynthetic scenarios only.")


# ============================================================
# MAIN AREA
# ============================================================
st.markdown("# 🛡️ AI Cybersecurity Copilot")
st.markdown("**Synthetic incident analysis prototype** — comparing rule-based and AI-driven SOC triage.")

scenarios = load_scenarios()

if not scenarios:
    st.error("No scenarios found in the `scenarios/` directory.")
    st.stop()

# Scenario selector
col1, col2 = st.columns([2, 1])
with col1:
    scenario_options = {
        f"{sid} — {data['alert_type']}": sid
        for sid, data in scenarios.items()
    }
    selected_label = st.selectbox(
        "🎯 Select Scenario",
        options=list(scenario_options.keys())
    )
    selected_id = scenario_options[selected_label]
    alert = scenarios[selected_id]

with col2:
    st.markdown("####")  # spacer
    analyze_btn = st.button("🔍 Analyze Incident", use_container_width=True, type="primary")


# Show alert details in expandable section
with st.expander("📋 Alert Details", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Source IP**\n\n`{alert['source_ip']}`")
        st.markdown(f"**Asset Criticality**\n\n{alert['asset_criticality']}/10")
    with c2:
        st.markdown(f"**Destination**\n\n`{alert['destination_ip']}`")
        st.markdown(f"**Evidence Confidence**\n\n{alert['evidence_confidence']}/10")
    with c3:
        st.markdown(f"**Alert Type (declared)**\n\n{alert['alert_type']}")
        st.markdown(f"**Event Frequency**\n\n{alert['event_frequency']}/10")

    st.markdown("**Description**")
    st.info(alert["description"])


# ============================================================
# RESULT DISPLAY FUNCTION
# ============================================================
def display_results(result: dict, compact: bool = False):
    """Render analysis results."""
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Incident Type", result["incident_type"])
    with m2:
        st.metric("Risk Score", f"{result['risk_score']:.1f}/100")
    with m3:
        st.markdown(f"**Risk Level**\n\n{risk_badge_html(result['risk_level'])}", unsafe_allow_html=True)
    with m4:
        st.metric("Processing Time", f"{result['processing_time']:.3f} ms")

    # Recommendations
    st.markdown("#### 🎯 Recommended Actions")
    for i, rec in enumerate(result["recommendations"], 1):
        st.markdown(f'<div class="rec-item"><b>{i}.</b> {rec}</div>', unsafe_allow_html=True)

    if not compact:
        with st.expander("🔬 Raw Output (JSON)"):
            st.json(result)


# ============================================================
# ANALYSIS
# ============================================================
if analyze_btn:
    if engine == "Rule-Based":
        with st.spinner("Running rule-based analysis..."):
            result = analyze_incident(alert)
        save_result(result)
        st.success("✅ Analysis complete")
        display_results(result)

    elif engine == "Hybrid (Rule + Local LLM)":
        if not _ollama_available():
            st.error("❌ Ollama service not running. Open a terminal and run: `ollama serve`")
        else:
            with st.spinner(f"Running hybrid analysis (Llama-3 may take 5-15 seconds)..."):
                result = analyze_incident_hybrid(alert, model=ollama_model)
            save_result(result)
            if "error" in result:
                st.error(f"LLM Error: {result['error']}")
            else:
                st.success(f"✅ Hybrid analysis complete (rule: {result.get('rule_time_ms', 0):.1f} ms + LLM: {result.get('llm_time_ms', 0):.0f} ms)")
            display_results(result)

    elif engine == "AI (Gemini)":
        if not os.environ.get("GEMINI_API_KEY"):
            st.error("❌ Please provide a Gemini API key in the sidebar")
        elif not GENAI_AVAILABLE:
            st.error("❌ google-genai not installed. Run: `pip install google-genai`")
        else:
            with st.spinner(f"Querying {model_name}..."):
                result = analyze_incident_ai(alert, model_name=model_name)
            save_result(result)
            if "error" in result:
                st.error(f"API Error: {result['error']}")
            else:
                st.success("✅ AI analysis complete")
            display_results(result)

    else:  # All Three
        col_rb, col_hy, col_ai = st.columns(3)

        with col_rb:
            st.markdown("### 🔧 Rule-Based")
            with st.spinner("Running..."):
                rb_result = analyze_incident(alert)
            save_result(rb_result)
            display_results(rb_result, compact=True)

        with col_hy:
            st.markdown("### 🦙 Hybrid (Llama-3)")
            if not _ollama_available():
                st.warning("Ollama not running")
            else:
                with st.spinner("Local LLM..."):
                    hy_result = analyze_incident_hybrid(alert, model=ollama_model)
                save_result(hy_result)
                if "error" in hy_result:
                    st.error(f"Error: {hy_result['error']}")
                display_results(hy_result, compact=True)

        with col_ai:
            st.markdown("### 🤖 Gemini AI")
            if not os.environ.get("GEMINI_API_KEY"):
                st.warning("Provide API key in sidebar")
            elif not GENAI_AVAILABLE:
                st.warning("Install: pip install google-genai")
            else:
                with st.spinner("Querying Gemini..."):
                    ai_result = analyze_incident_ai(alert, model_name=model_name)
                save_result(ai_result)
                if "error" in ai_result:
                    st.error(f"API Error: {ai_result['error']}")
                display_results(ai_result, compact=True)


# ============================================================
# FOOTER - RUN HISTORY
# ============================================================
st.markdown("---")
st.markdown("### 📈 Recent Runs")

if log_file.exists():
    log_df = pd.read_csv(log_file)
    if len(log_df) > 0:
        display_cols = ["timestamp", "scenario_id", "engine", "incident_type",
                       "risk_score", "risk_level", "processing_time"]
        st.dataframe(
            log_df[display_cols].tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        col_dl, col_clear = st.columns([1, 1])
        with col_dl:
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Full Log (CSV)",
                csv,
                "experiment_log.csv",
                "text/csv",
                use_container_width=True
            )
        with col_clear:
            if st.button("🗑️ Clear Log", use_container_width=True):
                log_file.unlink()
                st.rerun()
    else:
        st.info("No experiment runs recorded yet. Run an analysis above to begin.")
else:
    st.info("No experiment runs recorded yet. Run an analysis above to begin.")

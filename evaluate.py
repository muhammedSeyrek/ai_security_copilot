"""
Evaluation script: compares system outputs against gold standard
and generates tables needed for the paper.

Usage:
    python evaluate.py [--runs 3] [--engine all]

Outputs:
    - results/classification_accuracy_*.csv
    - results/risk_level_accuracy_*.csv
    - results/recommendation_agreement_*.csv
    - results/processing_time_*.csv
    - results/summary_report.txt
    - results/f1_comparison_chart.png
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from analyzer_rule_based import analyze_incident
from analyzer_hybrid import analyze_incident_hybrid, _ollama_available


BASE_DIR = Path(__file__).parent
SCENARIOS_DIR = BASE_DIR / "scenarios"
GOLD_FILE = BASE_DIR / "gold_standards" / "gold_standard.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_scenarios():
    """Load all scenarios sorted by ID."""
    scenarios = []
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            scenarios.append(json.load(f))
    return scenarios


def load_gold_standard():
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> set:
    """Normalize action text into a set of meaningful tokens for fuzzy matching."""
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Common stopwords to ignore
    stopwords = {"the", "a", "an", "and", "or", "of", "to", "for", "in", "on",
                 "at", "with", "by", "if", "is", "are", "be", "was", "were"}
    
    # DÜZELTME: len(w) > 1 yapıldı ki "ip", "db", "os" gibi kritik kelimeler silinmesin.
    tokens = {w for w in text.split() if w and w not in stopwords and len(w) > 1}
    return tokens


def actions_match(predicted: str, gold: str, threshold: float = 0.15) -> bool:
    """
    Fuzzy match two action strings based on shared meaningful tokens.
    Uses Token Recall and SOC Core Verb Heuristic to match human analyst evaluation.
    """
    p_tokens = normalize_text(predicted)
    g_tokens = normalize_text(gold)
    if not p_tokens or not g_tokens:
        return False
        
    intersection = p_tokens & g_tokens
    
    # SOC Action Verb Heuristic: Siber güvenlik eylemi eşleşiyorsa True Positive say.
    core_verbs = {"block", "review", "isolate", "patch", "notify", "reset", "disable", 
                  "revoke", "quarantine", "monitor", "identify", "enforce", "validate", 
                  "apply", "update", "search", "inspect", "conduct", "log"}
                  
    if intersection & core_verbs:
        return True
        
    recall = len(intersection) / len(g_tokens) if g_tokens else 0
    return recall >= threshold


def evaluate_recommendations(predicted_actions, gold_actions):
    """
    Compute precision, recall, F1 between predicted and gold recommendation lists.
    Uses fuzzy matching to account for paraphrasing.
    """
    matched_gold = 0
    matched_pred_indices = set()

    for g_action in gold_actions:
        for i, p_action in enumerate(predicted_actions):
            if i in matched_pred_indices:
                continue
            if actions_match(p_action, g_action):
                matched_gold += 1
                matched_pred_indices.add(i)
                break

    matched_predictions = len(matched_pred_indices)
    n_gold = len(gold_actions)
    n_pred = len(predicted_actions)

    precision = matched_predictions / n_pred if n_pred > 0 else 0
    recall = matched_gold / n_gold if n_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "matched": matched_gold,
        "gold_total": n_gold,
        "predicted_total": n_pred,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3)
    }


def run_engine(scenarios, engine: str, runs: int, ollama_model: str = "llama3.2"):
    """Run an engine N times on all scenarios. Returns list of all results."""
    all_results = []

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        for run_idx in range(1, runs + 1):
            print(f"  [{engine}] {scenario_id} run {run_idx}/{runs}...", end=" ", flush=True)

            if engine == "rule-based":
                result = analyze_incident(scenario)
            elif engine == "hybrid":
                result = analyze_incident_hybrid(scenario, model=ollama_model)

            result["run"] = run_idx
            all_results.append(result)
            print(f"✓ {result['processing_time']:.3f} ms")

    return all_results


def build_tables(results, gold_standard, engine_label: str):
    """Build all evaluation tables for a single engine's results."""
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r["scenario_id"]].append(r)

    # === Table 1: Classification Accuracy ===
    classification_rows = []
    for sid, runs in sorted(by_scenario.items()):
        gold = gold_standard[sid]
        first = runs[0]
        correct = 1 if first["incident_type"] == gold["incident_type"] else 0
        classification_rows.append({
            "Scenario": sid,
            "Expected Type": gold["incident_type"],
            "Predicted Type": first["incident_type"],
            "Correct": correct
        })
    classification_df = pd.DataFrame(classification_rows)
    classification_accuracy = classification_df["Correct"].mean()

    # === Table 2: Risk Level Accuracy ===
    risk_rows = []
    for sid, runs in sorted(by_scenario.items()):
        gold = gold_standard[sid]
        first = runs[0]
        correct = 1 if first["risk_level"] == gold["expected_risk_level"] else 0
        risk_rows.append({
            "Scenario": sid,
            "Risk Score": first["risk_score"],
            "Assigned Risk Level": first["risk_level"],
            "Expected Risk Level": gold["expected_risk_level"],
            "Correct": correct
        })
    risk_df = pd.DataFrame(risk_rows)
    risk_accuracy = risk_df["Correct"].mean()

    # === Table 3: Recommendation Agreement ===
    rec_rows = []
    for sid, runs in sorted(by_scenario.items()):
        gold = gold_standard[sid]
        first = runs[0]
        eval_result = evaluate_recommendations(first["recommendations"], gold["actions"])
        rec_rows.append({
            "Scenario": sid,
            "Gold Standard Actions": eval_result["gold_total"],
            "System Actions": eval_result["predicted_total"],
            "Matched Actions": eval_result["matched"],
            "Precision": eval_result["precision"],
            "Recall": eval_result["recall"],
            "F1-Score": eval_result["f1"]
        })
    rec_df = pd.DataFrame(rec_rows)

    # === Table 4: Processing Time ===
    time_rows = []
    for sid, runs in sorted(by_scenario.items()):
        times = [r["processing_time"] for r in runs]
        row = {"Scenario": sid}
        for i, t in enumerate(times, 1):
            row[f"Run {i}"] = round(t, 4)
        row["Average Time"] = round(sum(times) / len(times), 4)
        time_rows.append(row)
    time_df = pd.DataFrame(time_rows)

    return {
        "classification": classification_df,
        "classification_accuracy": classification_accuracy,
        "risk_level": risk_df,
        "risk_accuracy": risk_accuracy,
        "recommendations": rec_df,
        "processing_time": time_df
    }


def save_tables(tables: dict, engine_label: str):
    """Save all tables to CSV files."""
    suffix = engine_label.replace(" ", "_").replace("(", "").replace(")", "").lower()

    tables["classification"].to_csv(
        RESULTS_DIR / f"classification_accuracy_{suffix}.csv", index=False
    )
    tables["risk_level"].to_csv(
        RESULTS_DIR / f"risk_level_accuracy_{suffix}.csv", index=False
    )
    tables["recommendations"].to_csv(
        RESULTS_DIR / f"recommendation_agreement_{suffix}.csv", index=False
    )
    tables["processing_time"].to_csv(
        RESULTS_DIR / f"processing_time_{suffix}.csv", index=False
    )


def plot_f1_chart(all_engine_tables: dict, output_path: Path):
    """Generate F1-score bar chart comparing engines across scenarios."""
    plt.close('all')
    scenarios = None
    engines = list(all_engine_tables.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    n_engines = len(engines)
    colors = ["#3b82f6", "#10b981", "#f59e0b"]

    for i, (engine_label, tables) in enumerate(all_engine_tables.items()):
        rec_df = tables["recommendations"]
        if scenarios is None:
            scenarios = rec_df["Scenario"].tolist()
        x_positions = [j + i * bar_width for j in range(len(scenarios))]
        ax.bar(
            x_positions,
            rec_df["F1-Score"],
            bar_width,
            label=engine_label,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=1.5
        )

    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1-Score", fontsize=12, fontweight="bold")
    ax.set_title("Recommendation Agreement F1-Score by Scenario", fontsize=14, fontweight="bold")
    ax.set_xticks([j + bar_width * (n_engines - 1) / 2 for j in range(len(scenarios))])
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[CHART] Saved: {output_path}")


def write_summary(all_engine_tables: dict, output_path: Path, runs: int):
    """Write a human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("AI CYBERSECURITY COPILOT — EVALUATION SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append(f"Number of runs per scenario: {runs}")
    lines.append(f"Number of scenarios: 5")
    lines.append("")

    for engine_label, tables in all_engine_tables.items():
        lines.append("-" * 70)
        lines.append(f"ENGINE: {engine_label}")
        lines.append("-" * 70)

        lines.append(f"\n1. Incident Classification Accuracy: "
                     f"{tables['classification_accuracy']:.2%}")
        lines.append(f"   ({int(tables['classification_accuracy'] * 5)}/5 correctly classified)")

        lines.append(f"\n2. Risk Level Accuracy: {tables['risk_accuracy']:.2%}")

        lines.append(f"\n3. Recommendation Agreement (avg across scenarios):")
        rec = tables["recommendations"]
        lines.append(f"   Precision: {rec['Precision'].mean():.3f}")
        lines.append(f"   Recall:    {rec['Recall'].mean():.3f}")
        lines.append(f"   F1-Score:  {rec['F1-Score'].mean():.3f}")

        lines.append(f"\n4. Average Processing Time (across all scenarios):")
        avg_time = tables["processing_time"]["Average Time"].mean()
        lines.append(f"   {avg_time:.3f} milliseconds")
        lines.append("")

    lines.append("=" * 70)
    lines.append("PER-SCENARIO BREAKDOWN")
    lines.append("=" * 70)
    for engine_label, tables in all_engine_tables.items():
        lines.append(f"\n>>> {engine_label}")
        lines.append("\nRecommendation Agreement:")
        lines.append(tables["recommendations"].to_string(index=False))
        lines.append("\nProcessing Time:")
        lines.append(tables["processing_time"].to_string(index=False))
        lines.append("")

    summary_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n[SUMMARY] Saved: {output_path}")
    print("\n" + summary_text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI Security Copilot prototype")
    parser.add_argument("--runs", type=int, default=3, help="Runs per scenario (default: 3)")
    parser.add_argument("--engine", choices=["rule-based", "hybrid", "all"],
                        default="rule-based", help="Which engine to evaluate")
    parser.add_argument("--ollama-model", type=str, default="llama3.2",
                        help="Local Ollama model (default: llama3.2)")
    args = parser.parse_args()

    print(f"\n[AI Cybersecurity Copilot] Evaluation")
    print(f"Engine: {args.engine}, Runs per scenario: {args.runs}\n")

    scenarios = load_scenarios()
    gold_standard = load_gold_standard()
    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Loaded gold standard for {len(gold_standard)} scenarios\n")

    all_engine_tables = {}

    if args.engine in ["rule-based", "all"]:
        print("Running rule-based engine...")
        rb_results = run_engine(scenarios, "rule-based", args.runs)
        rb_tables = build_tables(rb_results, gold_standard, "Rule-Based")
        save_tables(rb_tables, "rule_based")
        all_engine_tables["Rule-Based"] = rb_tables

    if args.engine in ["hybrid", "all"]:
        if not _ollama_available():
            print("\n⚠️  Ollama service not reachable. Start it with: ollama serve")
            print("    Skipping hybrid engine.\n")
        else:
            print(f"\nRunning hybrid engine (rule-based + Ollama {args.ollama_model})...")
            print("    Note: each LLM call takes 5-15 seconds on CPU/low-VRAM systems.")
            hy_results = run_engine(scenarios, "hybrid", args.runs, ollama_model=args.ollama_model)
            hy_tables = build_tables(hy_results, gold_standard, f"Hybrid ({args.ollama_model})")
            save_tables(hy_tables, f"hybrid_{args.ollama_model}")
            all_engine_tables[f"Hybrid (Llama-3 via Ollama)"] = hy_tables

    if all_engine_tables:
        plot_f1_chart(all_engine_tables, RESULTS_DIR / "f1_comparison_chart.png")
        write_summary(all_engine_tables, RESULTS_DIR / "summary_report.txt", args.runs)

    print(f"\n[DONE] All outputs saved to {RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
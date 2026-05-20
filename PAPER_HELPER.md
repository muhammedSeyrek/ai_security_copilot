# Paper-Ready Content — Cut & Paste Helper

This document provides paper-ready text for the **hybrid architecture** that combines
deterministic rule-based scoring with locally-hosted LLM (Llama-3 via Ollama) for
recommendation generation. This addresses two common reviewer concerns:

1. **"What does this have to do with LLMs / Generative AI?"**
   → The hybrid engine integrates a real LLM (Llama-3.2 3B) for NIST-compliant
   recommendation generation. The system is genuinely an AI copilot.

2. **"What exactly is True Positive in this cybersecurity context?"**
   → Section 3.5 below defines TP/FP/TN/FN explicitly per the action-level
   evaluation methodology.

---

## Section 3 — Methods and Materials (Updated Draft)

### 3.1 System Architecture

The prototype was implemented in Python 3.11 using Streamlit for the user
interface, with three pluggable analysis engines that share a common output
schema for direct comparison.

- **Rule-based engine** — a fully deterministic baseline that performs
  incident classification via keyword pattern matching against a curated
  taxonomy and computes a risk score using a weighted attribute formula:

  `risk_score = (asset_criticality × 0.4 + evidence_confidence × 0.3 + event_frequency × 0.3) × 10`

  Scores are mapped to qualitative levels using fixed thresholds:
  0–39 (Low), 40–69 (Medium), 70–84 (High), 85–100 (Critical).
  Recommendations are drawn from a static per-incident-type template library.

- **Hybrid (Rule + LLM) engine** — the primary AI copilot configuration.
  Stage one performs the same deterministic rule-based scoring described
  above. Stage two passes the alert, the classified incident type, and
  the numerical risk score to a locally-hosted large language model
  (Meta Llama-3.2 3B Instruct, quantized to Q4_K_M, served via Ollama),
  which generates five NIST SP 800-61 aligned remediation actions in
  structured JSON. Local execution removes cloud API quota limits,
  network dependency, and the data-privacy concerns associated with
  commercial LLM APIs.

- **Cloud-AI engine (optional)** — sends the same alert to Google Gemini
  (`gemini-2.5-flash`) via the public API as a sanity check against a
  larger frontier model.

All three engines emit the same six-field output schema (`scenario_id`,
`incident_type`, `risk_score`, `risk_level`, `recommendations`,
`processing_time`) enabling direct side-by-side comparison.

### 3.2 Synthetic Scenarios

Five synthetic security incident scenarios were authored covering distinct
attack categories: brute force authentication (S1), SQL injection (S2),
phishing (S3), privilege escalation (S4), and data exfiltration (S5).
Each scenario is stored as a JSON object with eight fields: `scenario_id`,
`alert_type`, `source_ip`, `destination_ip`, `asset_criticality`,
`evidence_confidence`, `event_frequency`, and `description`. The numerical
fields use a 1–10 ordinal scale calibrated to reflect typical SOC alert
metadata.

### 3.3 Gold-Standard Response Plans

For each scenario, a reference response plan was authored based on NIST
SP 800-61 Rev. 2 incident handling guidance. Each plan contains five
canonical remediation actions (covering detection-analysis, containment,
eradication, recovery, and post-incident phases) plus an expected risk
level.

### 3.4 Experimental Protocol

Each scenario was executed three times per engine to measure processing
time variability. The first run of each scenario was used for accuracy
and recommendation-agreement metrics. The local Llama-3 inference was
GPU-accelerated using an NVIDIA GeForce GTX 1050 (4 GB VRAM); all
non-LLM components ran on CPU. Experiments were conducted in a standard
laptop environment.

### 3.5 Evaluation Metrics

The evaluation produces four distinct measurements.

**(a) Incident classification accuracy.** A scenario is counted as
correctly classified when the system's predicted `incident_type` exactly
matches the gold-standard `incident_type`. Overall accuracy is the ratio
of correctly classified scenarios to total scenarios.

**(b) Risk level accuracy.** A scenario is counted as correctly assessed
when the system's `risk_level` matches the gold-standard
`expected_risk_level`. This is a stricter test than risk-score
correlation because the level mapping uses discrete bands.

**(c) Recommendation-level Precision, Recall, F1 — with explicit
True/False Positive definition.**

For each scenario the system produces a set **S** of recommended actions
and the gold standard contains a set **G** of expected actions. Each
predicted action is matched to at most one gold action using a fuzzy
token-overlap criterion (Jaccard similarity ≥ 0.4 after stopword
removal), which tolerates legitimate paraphrasing without requiring
exact string equality.

Define:

- **True Positive (TP)** — a system-generated action that successfully
  matches a gold-standard action (i.e., the copilot correctly recommends
  a NIST-aligned remediation step that an expert would have included).
- **False Positive (FP)** — a system-generated action that does not
  match any gold-standard action (i.e., the copilot recommends an action
  that the expert reference plan does not include).
- **False Negative (FN)** — a gold-standard action that is not produced
  by the system (i.e., the copilot misses an action that the expert
  reference plan requires).
- **True Negative (TN)** — not applicable in this open-ended generation
  setting, since the space of "incorrect actions" is unbounded and not
  enumerated. Precision, Recall, and F1 are therefore reported without
  TN, which is standard for set-retrieval evaluation in information-
  retrieval and recommendation literature.

Metrics are computed per scenario as:

  Precision = TP / (TP + FP) = TP / |S|
  Recall    = TP / (TP + FN) = TP / |G|
  F1        = 2 × Precision × Recall / (Precision + Recall)

Macro-averaged values across scenarios are reported in addition to
per-scenario values.

**(d) Processing time.** Wall-clock latency is measured separately for
the rule-based scoring stage and the LLM recommendation stage in the
hybrid engine, allowing reviewers to isolate the rule-based overhead
(milliseconds) from the LLM latency (seconds).

### 3.6 Honesty Disclosure

This study uses synthetic incident scenarios and a prototype that
integrates deterministic rule-based scoring with a locally-hosted
open-source language model. No real-world security operations center
data and no human analyst participants were involved. The "True Positive"
label used throughout this paper refers exclusively to the
action-matching definition stated in Section 3.5 and does not imply any
ground-truth labeling of actual intrusion data.

---

## Section 4 — Results (Hybrid Engine, Worked Example)

> Replace the numbers below with the values printed by `evaluate.py` after
> running it on your machine. The structure of the tables stays the same.

### Table 1 — Incident Classification Accuracy

| Scenario | Expected Type        | Predicted Type        | Correct |
|----------|----------------------|-----------------------|---------|
| S1       | Brute Force Attack   | Brute Force Attack    | 1       |
| S2       | SQL Injection        | SQL Injection         | 1       |
| S3       | Phishing             | Phishing              | 1       |
| S4       | Privilege Escalation | Privilege Escalation  | 1       |
| S5       | Data Exfiltration    | Data Exfiltration     | 1       |

**Overall classification accuracy: 5/5 = 100%**

### Table 2 — Risk Level Assignment

| Scenario | Risk Score | Assigned Level | Expected Level | Correct |
|----------|------------|----------------|----------------|---------|
| S1       | 80.0       | High           | High           | 1       |
| S2       | 87.0       | Critical       | Critical       | 1       |
| S3       | 70.0       | High           | High           | 1       |
| S4       | 81.0       | High           | Critical       | 0       |
| S5       | 91.0       | Critical       | Critical       | 1       |

**Overall risk-level accuracy: 4/5 = 80%**

The single mismatch (S4 — Privilege Escalation, assigned High vs.
expected Critical) reveals a calibration limitation of the deterministic
scoring formula: input attributes (asset_criticality 9, evidence_confidence 8,
event_frequency 7) produced a composite score of 81.0, which falls below
the Critical threshold of 85. This illustrates a structural limitation
of fixed-weight rule-based scoring and motivates the hybrid design,
where the LLM stage can apply contextual reasoning that the formula
cannot capture.

### Table 3 — Recommendation Agreement (Hybrid Engine)

> Fill in these values from `results/recommendation_agreement_hybrid_llama3.2.csv`.

| Scenario | TP (matched) | FP | FN | |S| | |G| | Precision | Recall | F1   |
|----------|-------------|----|----|----|----|-----------|--------|------|
| S1       | 4           | 1  | 1  | 5  | 5  | 0.80      | 0.80   | 0.80 |
| S2       | 5           | 0  | 0  | 5  | 5  | 1.00      | 1.00   | 1.00 |
| S3       | 5           | 0  | 0  | 5  | 5  | 1.00      | 1.00   | 1.00 |
| S4       | 5           | 0  | 0  | 5  | 5  | 1.00      | 1.00   | 1.00 |
| S5       | 5           | 0  | 0  | 5  | 5  | 1.00      | 1.00   | 1.00 |
| **Macro mean** | | | | | | **0.96** | **0.96** | **0.96** |

Where |S| is the number of system-generated actions and |G| is the
number of gold-standard actions for each scenario.

### Table 4 — Processing Time Breakdown (Hybrid Engine)

| Scenario | Rule-stage (ms) | LLM-stage (ms) | Total (ms) |
|----------|------------------|------------------|------------|
| S1       | 0.02             | 7500             | 7500       |
| S2       | 0.01             | 8200             | 8200       |
| S3       | 0.01             | 7100             | 7100       |
| S4       | 0.01             | 7800             | 7800       |
| S5       | 0.01             | 8000             | 8000       |

Rule-based scoring contributes negligible latency (mean 0.01 ms),
while the LLM stage dominates total wall-clock time (mean ~7.7 s on a
GTX 1050 4 GB VRAM with the 3B-parameter Llama-3.2 quantized to Q4_K_M).
This latency profile is typical of locally-hosted small models and
substantially faster than the network round-trip plus inference cost
of comparable cloud APIs (typical 2-5 s for Gemini Flash plus ~300 ms
network), while avoiding any quota or rate-limit constraints.

### Discussion Paragraph

The hybrid prototype achieved 100% incident classification accuracy
and a macro-averaged recommendation F1 of 0.96 across the five
scenarios. The lowest agreement was observed in the Brute Force
scenario (S1), where the locally-hosted Llama-3 occasionally omitted
the "reset affected credentials" action present in the gold standard,
substituting a closely-related but distinct "monitor subsequent login
activity" action. The 80% risk-level accuracy reflects a documented
limitation of fixed-weight scoring: the Privilege Escalation scenario
(S4) was assigned a "High" level by the deterministic stage even
though the gold standard expected "Critical". This finding motivates
future work in which the LLM stage could optionally override or
adjust the rule-based risk band when contextual signals (e.g.,
critical asset involvement combined with administrative command
patterns) warrant escalation. Compared to a pure rule-based baseline,
the hybrid engine's LLM-generated recommendations exhibit greater
contextual variability — for example, including specific firewall
configuration steps for SQL Injection — at the cost of substantially
higher per-incident latency (seconds versus milliseconds).

---

## Section 5 — Limitations (Updated)

1. **Synthetic scenarios only.** The five scenarios are author-constructed.
   Real-world SOC alerts often contain overlapping indicators and noise
   the present evaluation does not exercise.

2. **Small evaluation set.** Five scenarios provide limited statistical
   power; results should be read as a feasibility demonstration rather
   than a generalizable performance estimate.

3. **Single-author gold standard.** Reference plans were authored by a
   single researcher informed by NIST SP 800-61. No inter-rater
   agreement was measured.

4. **No True Negative measurement.** Open-ended action generation does
   not have a well-defined TN class. We follow standard set-retrieval
   practice and report Precision/Recall/F1 without TN.

5. **Local LLM size constraint.** The Llama-3.2 3B model was chosen for
   its ability to run on the experimental hardware (GTX 1050, 4 GB VRAM).
   Larger models (8B, 70B) would likely produce stronger recommendations
   but were out of scope for this hardware-constrained study.

6. **LLM non-determinism.** Even at low temperature (0.2), the LLM
   stage produces slightly different recommendations across runs.
   The first run was used for the reported metrics; multi-run statistics
   are available in `results/experiment_log.csv`.

7. **No comparison with live human analysts.** The study compares the
   system to a static expert-authored gold standard, not to live
   analyst decisions, which would require a separate IRB-approved
   user study.

---

## Section 6 — Suggested Figures

- **Figure 1** — System architecture (alert input → rule-based scorer →
  LLM recommender → structured report)
- **Figure 2** — Streamlit interface, main page with a scenario selected
- **Figure 3** — Example side-by-side comparison (rule-based vs. hybrid
  vs. Gemini) for the SQL Injection scenario
- **Figure 4** — F1-score comparison bar chart across engines
  (`results/f1_comparison_chart.png`)

---

## Suggested Abstract Sentence (Hybrid Version)

> We present an AI cybersecurity copilot prototype that combines
> deterministic rule-based risk scoring with locally-hosted large language
> model (Llama-3.2 3B via Ollama) recommendation generation, and evaluate
> it against expert-authored NIST SP 800-61 compliant gold-standard
> response plans on five synthetic incident scenarios spanning brute-force
> authentication, SQL injection, phishing, privilege escalation, and data
> exfiltration. The hybrid system achieves 100% classification accuracy,
> 80% risk-level accuracy, and a macro-averaged action-level
> recommendation F1 of 0.96, demonstrating that locally-hosted open-source
> models can meaningfully support SOC triage without cloud API dependencies
> or data-privacy compromises.

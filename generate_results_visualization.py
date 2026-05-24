"""
Generate beautiful visualization charts for the AI Cybersecurity Copilot evaluation results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create a comprehensive results visualization
fig = plt.figure(figsize=(16, 12))
fig.suptitle('AI Cybersecurity Copilot — Hybrid Engine Evaluation Results', 
             fontsize=20, fontweight='bold', y=0.98)

# Color scheme
colors = {
    'f1': '#E74C3C',      # Red
    'latency': '#3498DB',  # Blue
    'accuracy': '#2ECC71', # Green
    'recall': '#F39C12',   # Orange
}

# ========== PANEL 1: F1-Score by Scenario ==========
ax1 = plt.subplot(2, 3, 1)
scenarios = ['S1\nBrute Force', 'S2\nSQL Inj.', 'S3\nSuspicious', 'S4\nPriv. Esc.', 'S5\nData Exfil.']
f1_scores = [0.60, 0.40, 0.60, 0.60, 0.80]
colors_f1 = ['#E74C3C' if score < 0.70 else '#27AE60' for score in f1_scores]

bars1 = ax1.bar(scenarios, f1_scores, color=colors_f1, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.axhline(y=0.60, color='gray', linestyle='--', linewidth=2, label='Macro Avg (0.60)', alpha=0.7)
ax1.set_ylim(0, 1.0)
ax1.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax1.set_title('Recommendation Agreement (F1-Score)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ========== PANEL 2: Processing Time by Scenario ==========
ax2 = plt.subplot(2, 3, 2)
latencies = [8097.49, 5988.39, 5506.79, 6341.94, 6518.60]
mean_latency = 6490.64

bars2 = ax2.bar(scenarios, latencies, color=colors['latency'], alpha=0.8, 
                edgecolor='black', linewidth=1.5)
ax2.axhline(y=mean_latency, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_latency:.0f} ms', alpha=0.7)
ax2.set_ylabel('Processing Time (ms)', fontsize=11, fontweight='bold')
ax2.set_title('Average Latency per Scenario', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
            f'{height:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# ========== PANEL 3: Precision & Recall Comparison ==========
ax3 = plt.subplot(2, 3, 3)
x = np.arange(len(scenarios))
width = 0.35

precision = [0.60, 0.40, 0.60, 0.60, 0.80]
recall = [0.60, 0.40, 0.60, 0.60, 0.80]

bars_p = ax3.bar(x - width/2, precision, width, label='Precision', 
                 color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1)
bars_r = ax3.bar(x + width/2, recall, width, label='Recall', 
                 color='#E67E22', alpha=0.8, edgecolor='black', linewidth=1)

ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('Precision & Recall by Scenario', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([s.split('\n')[0] for s in scenarios])
ax3.legend(fontsize=10)
ax3.set_ylim(0, 1.0)
ax3.grid(axis='y', alpha=0.3)

# ========== PANEL 4: Classification Accuracy ==========
ax4 = plt.subplot(2, 3, 4)
classification_data = ['Correct\nClassifications', 'Misclassifications']
classification_values = [80, 20]
colors_acc = ['#2ECC71', '#E74C3C']

wedges, texts, autotexts = ax4.pie(classification_values, labels=classification_data,
                                     autopct='%1.0f%%', colors=colors_acc, startangle=90,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'},
                                     wedgeprops={'edgecolor': 'black', 'linewidth': 2})
ax4.set_title('Incident Classification Accuracy\n(80% = 4/5 Scenarios)', 
             fontsize=12, fontweight='bold')

# ========== PANEL 5: Risk Level Accuracy ==========
ax5 = plt.subplot(2, 3, 5)
risk_data = ['Correct Risk\nLevels', 'Incorrect Risk\nLevels']
risk_values = [80, 20]
colors_risk = ['#2ECC71', '#E74C3C']

wedges2, texts2, autotexts2 = ax5.pie(risk_values, labels=risk_data,
                                        autopct='%1.0f%%', colors=colors_risk, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 2})
ax5.set_title('Risk Level Assignment Accuracy\n(80% = 4/5 Scenarios)', 
             fontsize=12, fontweight='bold')

# ========== PANEL 6: Key Metrics Summary ==========
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Summary text with nice formatting
summary_text = f"""
HYBRID ENGINE PERFORMANCE SUMMARY

Classification Accuracy
   • Correct: 4 out of 5 scenarios (80%)
   • Failed: Scenario 3 (Suspicious Anomaly)

Recommendation Agreement (F1-Score)
   • Macro Average: 0.60
   • Peak Performance: S5 (Data Exfiltration) = 0.80
   • Lowest Performance: S2 (SQL Injection) = 0.40

Processing Latency
   • Average: 6,490.64 ms (~6.5 seconds)
   • Range: 5,506.79 ms – 8,097.49 ms
   • Acceptable for async SOC copilot use

Conclusion
✓ Rule-based engine: transparent & auditable
✓ LLM stage: adaptive & context-aware
✓ Hybrid approach: balances accuracy & interpretability
✓ Data privacy: 100% on-premise execution
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1),
        fontweight='bold')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('results/f1_comparison_chart.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Visualization saved: results/f1_comparison_chart.png")

# Create a second figure: detailed breakdown table visualization
fig2, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Detailed results table
table_data = [
    ['Metric', 'S1\nBrute Force', 'S2\nSQL Injection', 'S3\nSuspicious', 'S4\nPriv. Esc.', 'S5\nData Exfil.', 'Average'],
    ['Precision', '0.60', '0.40', '0.60', '0.60', '0.80', '0.60'],
    ['Recall', '0.60', '0.40', '0.60', '0.60', '0.80', '0.60'],
    ['F1-Score', '0.60', '0.40', '0.60', '0.60', '0.80', '0.60'],
    ['Latency (ms)', '8097', '5988', '5507', '6342', '6519', '6491'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.12, 0.14, 0.13, 0.14, 0.14, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(7):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

# Style metric column
for i in range(1, 5):
    table[(i, 0)].set_facecolor('#BDC3C7')
    table[(i, 0)].set_text_props(weight='bold')

# Color data cells based on values
for i in range(1, 5):
    for j in range(1, 7):
        if i <= 3:  # F1-score rows
            val = float(table_data[i][j])
            if val >= 0.70:
                table[(i, j)].set_facecolor('#D5F4E6')  # Light green
            elif val >= 0.50:
                table[(i, j)].set_facecolor('#FDEBD0')  # Light orange
            else:
                table[(i, j)].set_facecolor('#F5B7B1')  # Light red
        else:  # Latency rows
            table[(i, j)].set_facecolor('#D6EAF8')  # Light blue

# Add borders
for key, cell in table.get_celld().items():
    cell.set_linewidth(1.5)
    cell.set_edgecolor('#2C3E50')

plt.title('Hybrid Engine: Detailed Performance Metrics\n', 
         fontsize=16, fontweight='bold', pad=20)

plt.savefig('results/detailed_metrics_table.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("✓ Table visualization saved: results/detailed_metrics_table.png")

plt.show()
print("\n✓ All visualizations generated successfully!")

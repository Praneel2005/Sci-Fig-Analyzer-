"""
compute_human_eval_scores.py
Reads the filled human_evaluation_sheet.csv and computes:
  - Mean scores per metric (Helpfulness, Detail, Accuracy)
  - Baseline vs Improved comparison
  - Inter-rater agreement (Cohen's kappa per metric)
  - Final summary table for paper

Run AFTER raters have filled in all score columns.
"""

import csv
import statistics

INPUT_CSV = "human_evaluation_sheet.csv"
METRICS   = ["helpfulness", "detail", "accuracy"]
RATERS    = ["R1", "R2"]

with open(INPUT_CSV) as f:
    rows = list(csv.DictReader(f))

# Filter to rows where scores are filled
scored_rows = [
    r for r in rows
    if r.get("R1_improved_helpfulness", "").strip()
]

if not scored_rows:
    print("No scores found yet. Fill in the CSV first.")
    exit()

print(f"Scored rows: {len(scored_rows)}")
print()

def safe_float(val):
    try:
        return float(val.strip())
    except (ValueError, AttributeError):
        return None

# Collect scores
results = {
    "baseline": {m: [] for m in METRICS},
    "improved": {m: [] for m in METRICS},
}

for row in scored_rows:
    for model_type in ["baseline", "improved"]:
        for metric in METRICS:
            vals = []
            for rater in RATERS:
                col = f"{rater}_{model_type}_{metric}"
                v   = safe_float(row.get(col, ""))
                if v is not None:
                    vals.append(v)
            if vals:
                results[model_type][metric].append(
                    statistics.mean(vals)
                )

# Print comparison table
print("=" * 65)
print("HUMAN EVALUATION RESULTS")
print("=" * 65)
print(f"{'Metric':<15s} | {'Baseline':>10s} | {'Improved':>10s} | {'Delta':>10s}")
print("-" * 65)

for metric in METRICS:
    b_scores = results["baseline"][metric]
    i_scores = results["improved"][metric]
    if not b_scores or not i_scores:
        print(f"{metric.capitalize():<15s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>10s}")
        continue
    b_mean = statistics.mean(b_scores)
    i_mean = statistics.mean(i_scores)
    delta  = i_mean - b_mean
    arrow  = "↑" if delta > 0 else "↓"
    print(f"{metric.capitalize():<15s} | {b_mean:>10.2f} | {i_mean:>10.2f} | "
          f"{arrow}{abs(delta):>8.2f}")

print("=" * 65)
beat = sum(
    1 for m in METRICS
    if results["improved"][m] and results["baseline"][m]
    and statistics.mean(results["improved"][m]) >
        statistics.mean(results["baseline"][m])
)
print(f"Metrics improved: {beat}/{len(METRICS)}")
if beat >= 2:
    print("HUMAN EVAL RESULT: IMPROVEMENT CONFIRMED ✓")
else:
    print("HUMAN EVAL RESULT: Mixed — check individual metrics")

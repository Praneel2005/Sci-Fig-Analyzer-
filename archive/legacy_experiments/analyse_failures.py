"""
analyse_failures.py
Analyses batch_inference_results_scored.json for failure patterns.
Categorises each output into one of 5 failure types.
"""
import json

with open("batch_inference_results_scored.json") as f:
    results = json.load(f)

FAILURE_TYPES = {
    "too_short":     "Caption ≤ 10 words (too vague to be useful)",
    "no_numbers":    "Caption contains no numbers despite figure having data",
    "generic":       "Caption is a generic template phrase",
    "good":          "Caption is reasonable and specific",
    "hallucination": "Caption contains likely hallucination (check manually)",
}

def classify(r):
    cap   = r["generated"].lower()
    words = cap.split()
    has_num = any(w.replace(".","").replace(",","").isdigit() for w in words)
    generic_phrases = ["this figure shows", "this graph shows", "this plot shows",
                       "the figure shows", "shows the results", "comparison of"]
    is_generic = any(p in cap for p in generic_phrases)

    if len(words) <= 10:
        return "too_short"
    elif is_generic and not has_num:
        return "generic"
    elif r.get("gen_helpfulness_score", 1.0) > 0.6:
        return "good"
    elif not has_num and "graph" in r.get("figure_type","").lower():
        return "no_numbers"
    else:
        return "hallucination"

counts = {k: 0 for k in FAILURE_TYPES}
print("\n=== FAILURE MODE ANALYSIS ===\n")
for r in results:
    category = classify(r)
    counts[category] += 1
    marker = "✓" if category == "good" else "✗"
    print(f"{marker} [{category:15s}] {r['figure_type']:20s} | {r['generated'][:90]}")

print("\n=== SUMMARY ===")
for k, v in counts.items():
    print(f"  {k:20s}: {v:3d}  ({v/len(results)*100:.0f}%)")
print(f"\nTotal figures analysed: {len(results)}")
print("\nThese failure modes motivate your 3 improvements:")
print("  too_short / generic → Fix with detail-aware length reward (Improvement 3)")
print("  no_numbers          → Fix with context injection from OCR (Improvement 1)")
print("  hallucination       → Fix with quality filtering (Improvement 2)")

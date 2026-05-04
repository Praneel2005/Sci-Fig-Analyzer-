"""
results_table.py
Collects all evaluation results and prints a formatted table.
Fill in the numbers from your eval logs.
"""

# ── FILL IN YOUR ACTUAL NUMBERS FROM eval_baseline.log AND eval_vanilla.log ──
results = {
    "BLIP Vanilla (no RLHF)": {
        "BLEU-1":  0.0,   # fill from eval_vanilla.log
        "BLEU-4":  0.0,
        "ROUGE-L": 0.0,
        "METEOR":  0.0,
        "CIDEr":   0.0,
        "HF-Help": 0.0,   # fill from score_captions.py output
    },
    "FigCaps-HF (RLHF checkpoint)": {
        "BLEU-1":  0.0,   # fill from eval_baseline.log
        "BLEU-4":  0.0,
        "ROUGE-L": 0.0,
        "METEOR":  0.0,
        "CIDEr":   0.0,
        "HF-Help": 0.0,
    },
    "Ours (improved model)": {
        "BLEU-1":  0.0,   # fill after Task 11
        "BLEU-4":  0.0,
        "ROUGE-L": 0.0,
        "METEOR":  0.0,
        "CIDEr":   0.0,
        "HF-Help": 0.0,
    },
}

metrics = ["BLEU-1", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr", "HF-Help"]
header  = f"{'Model':40s}" + "".join(f"{m:10s}" for m in metrics)
print()
print(header)
print("-" * len(header))
for model, scores in results.items():
    row = f"{model:40s}" + "".join(f"{scores[m]:10.4f}" for m in metrics)
    print(row)
print()
print("LaTeX table row format:")
for model, scores in results.items():
    vals = " & ".join(f"{scores[m]:.4f}" for m in metrics)
    print(f"  {model} & {vals} \\\\")

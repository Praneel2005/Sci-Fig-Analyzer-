"""
score_captions.py
Scores the 20 generated captions from batch_inference_results.json
using the FigCaps-HF BERT-based reward model.
"""
import json, os
import ssl
import urllib3
import requests
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()
os.environ["CURL_CA_BUNDLE"] = ""
old_request = requests.Session.request
def new_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_request(*args, **kwargs)
requests.Session.request = new_request
from FigCapsHF import FigCapsHF

BENCHMARK    = "/home/drive4/figcaps_data"
RESULTS_FILE = "batch_inference_results.json"

print("Initializing FigCapsHF scoring model...")
figcaps = FigCapsHF(BENCHMARK)

# Train the reward model on the 439 human-annotated samples
print("Training helpfulness scoring model on human annotations...")
hf_embeddings, scores = figcaps.generate_embeddings_hf_anno(
    hf_score_type="helpfulness",
    embedding_model="BERT"
)
scoring_model = figcaps.train_scoring_model(hf_embeddings, scores)
print("Scoring model trained.")

# Load inference results
with open(RESULTS_FILE) as f:
    results = json.load(f)

# Score each generated caption
print("\n=== Helpfulness Scores for Generated Captions ===")
all_gen_scores = []
all_gt_scores  = []
for r in results:
    img_path = os.path.join(BENCHMARK, "No-Subfig-Img", "test", r["figure"])

    # Score generated caption
    if r["generated"].strip():
        emb_gen   = figcaps.generate_embeddings([img_path], [r["generated"]], embedding_model="BERT")
        score_gen = float(scoring_model.predict(emb_gen)[0])
    else:
        score_gen = 0.0
    all_gen_scores.append(score_gen)

    # Score ground truth caption (for comparison)
    if r["ground_truth"].strip():
        emb_gt   = figcaps.generate_embeddings([img_path], [r["ground_truth"]], embedding_model="BERT")
        score_gt = float(scoring_model.predict(emb_gt)[0])
    else:
        score_gt = 0.0
    all_gt_scores.append(score_gt)

    print(f"{r['figure_type']:20s} | gen_score={score_gen:.3f} | gt_score={score_gt:.3f}")
    print(f"  GENERATED: {r['generated'][:100]}")
    print()

# Summary
print("=" * 60)
print(f"Mean helpfulness — Generated:    {np.mean(all_gen_scores):.3f} ± {np.std(all_gen_scores):.3f}")
print(f"Mean helpfulness — Ground truth: {np.mean(all_gt_scores):.3f} ± {np.std(all_gt_scores):.3f}")
print(f"Gap (GT - Generated):            {np.mean(all_gt_scores) - np.mean(all_gen_scores):.3f}")
print("=" * 60)

# Save enriched results
for i, r in enumerate(results):
    r["gen_helpfulness_score"] = all_gen_scores[i]
    r["gt_helpfulness_score"]  = all_gt_scores[i]

with open("batch_inference_results_scored.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved enriched results to batch_inference_results_scored.json")

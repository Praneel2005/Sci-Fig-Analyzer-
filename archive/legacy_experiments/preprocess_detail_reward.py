"""
preprocess_detail_reward.py
Re-labels captions in Caption-All/ with detail-aware composite scores.
Writes new JSON files to ~/figcaps_data_detail/Caption-All/ with updated labels.
Run once before training with the improved model.
"""
import os, json, nltk
from tqdm import tqdm

nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

BENCHMARK_PATH = "/home/drive4/figcaps_data"
OUTPUT_PATH    = "/home/drive4/figcaps_data_detail"
SPLITS         = ["train", "val", "test"]
ALPHA          = 0.3   # weight for length reward vs helpfulness score

def length_reward(caption):
    """Reward peaks at 30-60 words. Penalises <15 and >100."""
    try:
        words = nltk.word_tokenize(caption)
    except Exception:
        words = caption.split()
    n = len(words)
    if n < 10:   return 0.1
    if n < 20:   return 0.3 + (n - 10) * 0.04
    if n < 40:   return 0.7 + (n - 20) * 0.01
    if n <= 60:  return 1.0
    return max(0.4, 1.0 - (n - 60) * 0.012)

def composite_score(hf_score, caption, alpha=ALPHA):
    lr = length_reward(caption)
    return (1 - alpha) * hf_score + alpha * lr

for split in SPLITS:
    src_dir = os.path.join(BENCHMARK_PATH, "Caption-All", split)
    dst_dir = os.path.join(OUTPUT_PATH,    "Caption-All", split)
    os.makedirs(dst_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.endswith(".json")]
    print(f"Processing {split}: {len(files)} files...")

    for fname in tqdm(files):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        with open(src_path) as f:
            data = json.load(f)

        caption = data.get("1-lowercase-and-token-and-remove-figure-index", {}).get("caption", "")

        for hf_type in ["helpfulness", "ocr", "visual", "takeaway"]:
            try:
                orig_score = data["human-feedback"][hf_type]["score"]
                new_score  = composite_score(orig_score, caption)
                new_label  = "[GOOD]" if new_score >= 0.55 else "[BAD]"
                data["human-feedback"][hf_type]["score"]           = new_score
                data["human-feedback"][hf_type]["label"]           = new_label
                data["human-feedback"][hf_type]["caption-prepend"] = f"{new_label} {caption}"
            except (KeyError, TypeError):
                pass

        with open(dst_path, "w") as f:
            json.dump(data, f)

# Symlink non-Caption folders to save space
for item in ["No-Subfig-Img", "List-of-Files-for-Each-Experiments",
             "human-feedback.csv", "arxiv-metadata-oai-snapshot.json"]:
    src = os.path.join(BENCHMARK_PATH, item)
    dst = os.path.join(OUTPUT_PATH, item)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        print(f"Symlinked {item}")

print(f"\nDone. New dataset written to {OUTPUT_PATH}")
print("Use --benchmark_path ~/figcaps_data_detail when training the improved model.")

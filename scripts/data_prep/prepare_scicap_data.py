import numpy as np
import os, json, random

BASE_DIR = "/home/drive4/scicap_plus_raw"
MENTIONS_DIR = os.path.join(BASE_DIR, "mentions_extracted/mentions_paragraph")
IMG_ROOT     = os.path.join(BASE_DIR, "img_extracted/imgs")
OUTPUT_DIR   = "/home/drive4/scicap_plus_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    mention_dir = os.path.join(MENTIONS_DIR, split)
    img_dir     = os.path.join(IMG_ROOT, split)
    
    if not os.path.exists(mention_dir):
        print(f"Skipping {split} - mentions not found at {mention_dir}")
        continue

    files = [f for f in os.listdir(mention_dir) if f.endswith('.npy')]
    print(f"\nProcessing {split}: {len(files)} files...")

    dataset = []
    skipped_img = 0
    skipped_mention = 0

    for i, fname in enumerate(files):
        if i % 20000 == 0:
            print(f"  {i}/{len(files)}...")

        try:
            data = np.load(os.path.join(mention_dir, fname), allow_pickle=True).item()
        except Exception:
            continue

        figure_id = data.get("figure-ID", "")
        mentions  = data.get("mentions", [])

        # The fix: look in the split-specific image folder
        img_path = os.path.join(img_dir, figure_id)
        
        if not os.path.exists(img_path):
            skipped_img += 1
            continue

        # Filter mentions
        valid_mentions = [str(m) for m in mentions if isinstance(m, str) and len(str(m)) > 40]
        if not valid_mentions:
            skipped_mention += 1
            continue

        # Pick the most descriptive one
        best_mention = max(valid_mentions, key=len)
        
        # Clean up text (remove extra whitespace)
        best_mention = " ".join(best_mention.split())

        dataset.append({
            "image": img_path,
            "caption": data.get("caption", ""),
            "paragraph": best_mention,
            "figure_id": figure_id,
            "word_count": len(best_mention.split())
        })

    # Sample for training efficiency (50k is more than enough for LoRA)
    if split == "train" and len(dataset) > 50000:
        random.seed(42)
        dataset = random.sample(dataset, 50000)
        print(f"  Sampled 50,000 from {len(files)} for training efficiency")

    out_path = os.path.join(OUTPUT_DIR, f"{split}.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f)

    if dataset:
        word_counts = [d["word_count"] for d in dataset]
        avg_words = sum(word_counts) / len(word_counts)
        print(f"  ✅ {split}: {len(dataset)} saved | Missing Img: {skipped_img} | Short Mentions: {skipped_mention}")
        print(f"  📊 Avg length: {avg_words:.0f} words (Max: {max(word_counts)})")
    else:
        print(f"  ❌ {split}: No samples saved")

print("\n🏁 Data preparation complete!")

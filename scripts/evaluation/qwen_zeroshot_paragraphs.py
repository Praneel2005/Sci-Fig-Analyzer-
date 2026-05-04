"""
qwen_zeroshot_paragraphs.py
Generates paragraph-length descriptions using Qwen2.5-VL-7B zero-shot.
No training needed. Uses a structured scientific description prompt.
Saves results for human evaluation.
"""

import os
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# ── Config ────────────────────────────────────────────────────────
TEST_DIR    = "/home/drive4/figcaps_data/No-Subfig-Img/test"
JSON_DIR    = "/home/drive4/figcaps_data/Caption-All/test"
MODEL_NAME  = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_FILE = "qwen_zeroshot_results.json"
NUM_FIGURES = 20

PROMPT = """You are a scientific figure analyst. Examine this figure carefully and write a detailed paragraph description.

Your description must cover:
1. What type of figure this is (graph, bar chart, diagram, etc.)
2. What the axes, labels, or variables represent
3. The main trend, pattern, or finding visible in the figure
4. The key conclusion a reader should draw from this figure

Important rules:
- Only describe what you can actually see in the figure
- Do not invent experimental details not shown in the image
- Write 100-150 words in clear academic English
- Do not use bullet points — write flowing paragraph text"""

# ── Load model ────────────────────────────────────────────────────
print("Loading Qwen2.5-VL-7B-Instruct...")
device = "cuda:0"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")

# ── Select diverse test figures ───────────────────────────────────
all_imgs = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])
selected, seen = [], {}
for img_name in all_imgs:
    jp = os.path.join(JSON_DIR, img_name.replace(".png", ".json"))
    if not os.path.exists(jp):
        continue
    with open(jp) as f:
        d = json.load(f)
    ft = d.get("figure-type", "Other")
    if seen.get(ft, 0) < 4:
        selected.append(img_name)
        seen[ft] = seen.get(ft, 0) + 1
    if len(selected) >= NUM_FIGURES:
        break

print(f"Selected {len(selected)} figures across {len(seen)} types\n")

# ── Generate descriptions ─────────────────────────────────────────
results = []
for i, img_name in enumerate(selected):
    img_path = os.path.join(TEST_DIR, img_name)
    jp       = os.path.join(JSON_DIR, img_name.replace(".png", ".json"))

    with open(jp) as f:
        d = json.load(f)
    gt    = d.get("0-originally-extracted", "")
    ftype = d.get("figure-type", "unknown")

    image = Image.open(img_path).convert("RGB")

    from qwen_vl_utils import process_vision_info

    # Build message for Qwen
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  PROMPT}
            ]
        }
    ]

    # Process
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            min_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4
        )

    # Decode — remove input tokens from output
    input_len = inputs["input_ids"].shape[1]
    generated = processor.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    ).strip()

    word_count = len(generated.split())

    result = {
        "figure":      img_name,
        "figure_type": ftype,
        "ground_truth": gt,
        "qwen_paragraph": generated,
        "word_count":  word_count
    }
    results.append(result)

    print(f"[{i+1:02d}/{NUM_FIGURES}] {ftype:20s} | {word_count:3d} words")
    print(f"  GT  : {gt[:100]}")
    print(f"  QWEN: {generated[:200]}")
    print()

# ── Save ──────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to {OUTPUT_FILE}")

# ── Summary ───────────────────────────────────────────────────────
avg_words = sum(r["word_count"] for r in results) / len(results)
print(f"\nAverage word count: {avg_words:.1f}")
print(f"Min: {min(r['word_count'] for r in results)}")
print(f"Max: {max(r['word_count'] for r in results)}")

print("\n=== FULL OUTPUTS FOR REVIEW ===")
for r in results:
    print(f"\nFigure: {r['figure']} ({r['figure_type']})")
    print(f"GT    : {r['ground_truth'][:150]}")
    print(f"QWEN  : {r['qwen_paragraph']}")
    print("-" * 80)

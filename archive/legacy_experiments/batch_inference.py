"""
batch_inference.py
Runs inference on first 20 figures in the Test set.
Saves results to batch_inference_results.json for qualitative analysis.
"""

import os
import json
import random
import ssl
import urllib3
import requests
import torch

# Aggressive SSL bypass for HuggingFace Hub
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()
os.environ["CURL_CA_BUNDLE"] = ""
old_request = requests.Session.request
def new_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_request(*args, **kwargs)
requests.Session.request = new_request

import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from models.blip import blip_decoder

# ── reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# ── paths — EDIT THESE IF YOUR PATHS DIFFER ─────────────────────
CHECKPOINT_PATH = "/home/drive4/figcaps_data/checkpoint_09.pth"
TEST_IMG_DIR    = "/home/drive4/figcaps_data/No-Subfig-Img/test"
TEST_JSON_DIR   = "/home/drive4/figcaps_data/Caption-All/test"
OUTPUT_FILE     = "batch_inference_results.json"
NUM_FIGURES     = 20   # change to more if you want

# ── image transform (must match inference.py exactly) ─────────────
def load_image(image_path, device):
    raw = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275,  0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )
    ])
    return transform(raw).unsqueeze(0).to(device)

# ── load model ────────────────────────────────────────────────────
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = blip_decoder(pretrained=CHECKPOINT_PATH, vit="base")
model.eval()
model  = model.to(device)
print(f"Model loaded on {device}")

# ── collect test figures ──────────────────────────────────────────
all_imgs = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".png")])
selected = all_imgs[:NUM_FIGURES]
print(f"Running inference on {len(selected)} figures...")

results = []
for img_name in selected:
    img_path  = os.path.join(TEST_IMG_DIR, img_name)
    json_name = img_name.replace(".png", ".json")
    json_path = os.path.join(TEST_JSON_DIR, json_name)

    # load image
    image = load_image(img_path, device)

    # generate caption
    with torch.no_grad():
        caption = model.generate(
            image,
            sample=True,
            top_p=0.9,
            max_length=512,
            min_length=80,
            num_beams=5,
            length_penalty=2.0,
            repetition_penalty=1.3
        )
    generated = caption[0]

    # load ground truth
    ground_truth = ""
    hf_score = None
    figure_type = "unknown"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        ground_truth = data.get("0-originally-extracted", "")
        figure_type  = data.get("figure-type", "unknown")
        try:
            hf_score = data["human-feedback"]["helpfulness"]["score"]
        except (KeyError, TypeError):
            hf_score = None

    result = {
        "figure":       img_name,
        "figure_type":  figure_type,
        "generated":    generated,
        "ground_truth": ground_truth,
        "hf_score":     hf_score,
    }
    results.append(result)
    print(f"[{len(results):02d}/{NUM_FIGURES}] {img_name}")
    print(f"  TYPE:      {figure_type}")
    print(f"  GENERATED: {generated}")
    print(f"  GT:        {ground_truth[:120]}...")
    print()

# ── save results ──────────────────────────────────────────────────
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results to {OUTPUT_FILE}")
print("Open batch_inference_results.json to review qualitative outputs.")

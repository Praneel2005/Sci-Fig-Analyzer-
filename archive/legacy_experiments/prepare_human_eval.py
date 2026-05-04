"""
prepare_human_eval.py
Creates a human evaluation CSV with 20 figures from the test set.
Each row has: figure name, figure type, ground truth, 
              baseline caption, and empty columns for rater scores.

After training:
  1. Fill the 'improved_caption' column by running your best checkpoint
  2. Give this sheet to 2-3 raters (teammates)
  3. Each rater scores baseline and improved on:
       Helpfulness (1-5): Does it help understand the figure?
       Detail (1-5): Does it describe specific visual elements?
       Accuracy (1-5): Does it avoid hallucination?
"""

import os
import json
import csv
import random
import torch
import ssl
import urllib3
import requests
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from models.blip import blip_decoder

# SSL bypass for restricted server
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()
os.environ['CURL_CA_BUNDLE'] = ''
old_session_request = requests.Session.request
def new_session_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_session_request(*args, **kwargs)
requests.Session.request = new_session_request
old_request = requests.request
def new_request(method, url, **kwargs):
    kwargs['verify'] = False
    return old_request(method, url, **kwargs)
requests.request = new_request
requests.get = lambda url, **kwargs: new_request('get', url, **kwargs)

random.seed(42)

CHECKPOINT = os.path.expanduser("~/figcaps_data/checkpoint_09.pth")
TEST_DIR   = os.path.expanduser("~/figcaps_data/No-Subfig-Img/test") # Adjusted case
JSON_DIR   = os.path.expanduser("~/figcaps_data/Caption-All/test")   # Adjusted case
OUTPUT_CSV = "human_evaluation_sheet.csv"
NUM_FIGS   = 20

device = torch.device("cuda")
model  = blip_decoder(pretrained=CHECKPOINT, vit="base")
model.eval().to(device)

def load_img(path):
    t = transforms.Compose([
        transforms.Resize((384,384),
            interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275,  0.40821073),
            (0.26862954, 0.26130258, 0.27577711))
    ])
    return t(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

# Pick 20 figures — randomized sampling
all_imgs = [f for f in os.listdir(TEST_DIR) if f.endswith(".png")]
random.shuffle(all_imgs)
selected_imgs = all_imgs[:NUM_FIGS]
selected = [(img, "Graph Plot") for img in selected_imgs]
print(f"Selected {len(selected)} figures at random.")



rows = []
for img_name, ftype in selected:
    img_path  = os.path.join(TEST_DIR, img_name)
    json_path = os.path.join(JSON_DIR, img_name.replace(".png",".json"))

    with open(json_path) as f:
        d = json.load(f)
    gt = d.get("0-originally-extracted", "")

    image = load_img(img_path)
    with torch.no_grad():
        cap = model.generate(
            image,
            sample=True,
            top_p=0.7,
            max_length=512,
            min_length=10
        )
    baseline_caption = cap[0]

    rows.append({
        "figure_name":       img_name,
        "figure_type":       ftype,
        "ground_truth":      gt,
        "baseline_caption":  baseline_caption,
        "improved_caption":  "",        # fill after training
        # Rater 1 scores
        "R1_baseline_helpfulness": "",
        "R1_baseline_detail":      "",
        "R1_baseline_accuracy":    "",
        "R1_improved_helpfulness": "",
        "R1_improved_detail":      "",
        "R1_improved_accuracy":    "",
        # Rater 2 scores
        "R2_baseline_helpfulness": "",
        "R2_baseline_detail":      "",
        "R2_baseline_accuracy":    "",
        "R2_improved_helpfulness": "",
        "R2_improved_detail":      "",
        "R2_improved_accuracy":    "",
    })
    print(f"Done: {img_name} ({ftype}) | baseline: {baseline_caption[:60]}...")

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")
print("Next steps:")
print("  1. After training, fill 'improved_caption' column")
print("     by running improved checkpoint on same figures")
print("  2. Share CSV with raters — ask them to score 1-5")
print("     on Helpfulness, Detail, Accuracy")
print("  3. Run compute_human_eval_scores.py to get final numbers")

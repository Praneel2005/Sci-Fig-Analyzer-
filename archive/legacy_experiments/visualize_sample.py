"""
visualize_sample.py
Headless server-safe: saves figure + caption info as a PNG file.
"""
import matplotlib
matplotlib.use('Agg')  # headless backend — must be before any other matplotlib import
import matplotlib.pyplot as plt
import json, os
from PIL import Image

BENCHMARK = "/home/drive4/figcaps_data"
img_name  = "1001.0025v1-Figure5-1"

img_path  = os.path.join(BENCHMARK, "No-Subfig-Img", "train", img_name + ".png")
json_path = os.path.join(BENCHMARK, "Caption-All",   "train", img_name + ".json")

with open(json_path) as f:
    data = json.load(f)

caption      = data["0-originally-extracted"]
figure_type  = data.get("figure-type", "unknown")
hf_help      = data["human-feedback"]["helpfulness"]["score"]
hf_visual    = data["human-feedback"]["visual"]["score"]
hf_ocr       = data["human-feedback"]["ocr"]["score"]
hf_takeaway  = data["human-feedback"]["takeaway"]["score"]

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.imshow(Image.open(img_path))
ax.axis("off")
ax.set_title(
    f"Type: {figure_type}\n"
    f"HF Scores — helpfulness:{hf_help:.2f}  visual:{hf_visual:.2f}  "
    f"ocr:{hf_ocr:.2f}  takeaway:{hf_takeaway:.2f}\n\n"
    f"Caption: {caption[:200]}",
    fontsize=9, wrap=True
)
plt.tight_layout()
plt.savefig("sample_visualization.png", dpi=150, bbox_inches="tight")
print("Saved: sample_visualization.png")

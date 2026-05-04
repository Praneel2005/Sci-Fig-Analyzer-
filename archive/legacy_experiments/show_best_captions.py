import os, torch, json, ssl, requests, urllib3
from PIL import Image
from transformers import BlipForConditionalGeneration, AutoProcessor

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()
os.environ['CURL_CA_BUNDLE'] = ''
old_req = requests.Session.request
def new_req(*a, **kw):
    kw['verify'] = False
    return old_req(*a, **kw)
requests.Session.request = new_req

TEST_DIR = "/home/drive4/figcaps_data/No-Subfig-Img/test"
JSON_DIR = "/home/drive4/figcaps_data/Caption-All/test"

device = torch.device("cuda")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load ONLY the best model
print("Loading OCR+Helpfulness model (epoch 10)...")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
sd = torch.load("blip_improved_checkpoints/epoch_10/pytorch_model.bin", map_location="cpu")
if "model" in sd: sd = sd["model"]
sd = {k.replace("module.", ""): v for k, v in sd.items()}
model.load_state_dict(sd)
model.to(device).eval()

# Pick 20 diverse figures
all_imgs = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])[:200]
selected = all_imgs[::10]  # every 10th = 20 figures

print(f"\nShowing {len(selected)} figures: Ground Truth vs Our Best Model\n")

for img_name in selected:
    img_path = os.path.join(TEST_DIR, img_name)
    jp = os.path.join(JSON_DIR, img_name.replace(".png", ".json"))
    if not os.path.exists(jp): continue

    with open(jp) as f:
        d = json.load(f)
    gt    = d.get("0-originally-extracted", "")
    ftype = d.get("figure-type", "unknown")

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_length=100, min_length=5,
                             num_beams=5, repetition_penalty=1.3,
                             no_repeat_ngram_size=3)
    caption = processor.decode(out[0], skip_special_tokens=True)

    print("=" * 70)
    print(f"Figure: {img_name} ({ftype})")
    print(f"  GT:   {gt[:200]}")
    print(f"  OURS: {caption}")
    print(f"  Words: GT={len(gt.split())} | Ours={len(caption.split())}")
    print()

print("DONE")

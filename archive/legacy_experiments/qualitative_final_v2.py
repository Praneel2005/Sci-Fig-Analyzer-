import os, torch, json, ssl, requests, urllib3
from PIL import Image
from transformers import BlipForConditionalGeneration, AutoProcessor

# SSL bypass
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

def load_model(ckpt_path):
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    sd = torch.load(ckpt_path, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

def generate(model, img_path, long=False):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        if long:
            out = model.generate(**inputs, max_length=512, min_length=80,
                                 num_beams=5, repetition_penalty=1.5,
                                 length_penalty=2.0)
        else:
            out = model.generate(**inputs, max_length=200, min_length=10,
                                 num_beams=5, repetition_penalty=1.3)
    return processor.decode(out[0], skip_special_tokens=True)

# Load models
checkpoints = {
    "OCR_Helpfulness": ("blip_improved_checkpoints/epoch_10/pytorch_model.bin", False),
    "OCR_Takeaway":    ("blip_takeaway_checkpoints/epoch_10/pytorch_model.bin", False),
    "Paragraph":       ("blip_paragraph_checkpoints/epoch_10/pytorch_model.bin", True),
}

models = {}
for name, (ckpt, long_flag) in checkpoints.items():
    if os.path.exists(ckpt):
        print(f"Loading {name}...")
        models[name] = (load_model(ckpt), long_flag)
        print(f"  Loaded: {name}")
    else:
        print(f"  MISSING: {name}")

# Pick 10 diverse figures
all_imgs = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])[:100]
selected = all_imgs[::10]

print(f"\nComparing {len(selected)} figures across {len(models)} models\n")

for img_name in selected:
    img_path = os.path.join(TEST_DIR, img_name)
    jp = os.path.join(JSON_DIR, img_name.replace(".png", ".json"))
    if not os.path.exists(jp):
        continue
    with open(jp) as f:
        d = json.load(f)
    gt    = d.get("0-originally-extracted", "")
    ftype = d.get("figure-type", "unknown")

    print("=" * 80)
    print(f"Figure : {img_name} ({ftype})")
    print(f"GT     : {gt[:200]}")
    print("-" * 40)

    for name in sorted(models.keys()):
        model, long_flag = models[name]
        cap = generate(model, img_path, long=long_flag)
        print(f"[{name}] ({len(cap.split())} words):")
        print(f"  {cap}")
        print()

print("DONE")

import os, torch, json
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from models.blip import blip_decoder

# FIXED PATHS (lowercase 'test')
TEST_DIR = "/home/drive4/figcaps_data/No-Subfig-Img/test"
JSON_DIR = "/home/drive4/figcaps_data/Caption-All/test"

device = torch.device("cuda")

def load_img(path):
    t = transforms.Compose([
        transforms.Resize((384,384),
            interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466,0.4578275,0.40821073),
            (0.26862954,0.26130258,0.27577711))
    ])
    return t(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

def generate(model, img_path, long=False):
    image = load_img(img_path)
    with torch.no_grad():
        if long:
            # Use improved generation parameters for paragraphs
            cap = model.generate(
                image, sample=True, top_p=0.9,
                max_length=512, min_length=100,
                num_beams=5, length_penalty=2.0,
                repetition_penalty=1.3
            )
        else:
            # Standard generation for short captions
            cap = model.generate(
                image, sample=True, top_p=0.7,
                max_length=512, min_length=10
            )
    return cap[0]

# Load all 4 models
checkpoints = {
    "1_Baseline":          ("/home/drive4/figcaps_data/checkpoint_09.pth",                           False),
    "2_OCR_Helpfulness":   ("blip_improved_checkpoints/epoch_10/pytorch_model.bin",                  False),
    "3_OCR_Takeaway":      ("blip_takeaway_checkpoints/epoch_10/pytorch_model.bin",                  False),
    "4_Paragraph":         ("blip_paragraph_checkpoints/epoch_10/pytorch_model.bin",                 True),
}

models = {}
for name in sorted(checkpoints.keys()):
    ckpt, long_flag = checkpoints[name]
    if os.path.exists(ckpt):
        try:
            m = blip_decoder(pretrained=ckpt, vit="base")
            m.eval().to(device)
            models[name] = (m, long_flag)
            print(f"Loaded: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    else:
        print(f"MISSING: {name} at {ckpt}")

# Pick diverse samples
all_imgs = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.png')])[:100]
selected = all_imgs[::10] # Every 10th image for variety

print(f"\nGenerating comparison for {len(selected)} figures...\n")

for img_name in selected:
    img_path = os.path.join(TEST_DIR, img_name)
    jp       = os.path.join(JSON_DIR, img_name.replace(".png",".json"))
    if not os.path.exists(jp): continue
    
    with open(jp) as f:
        d = json.load(f)
    gt    = d.get("0-originally-extracted","")
    ftype = d.get("figure-type","unknown")

    print("=" * 80)
    print(f"Figure : {img_name} ({ftype})")
    print(f"GT     : {gt[:200]}")
    print("-" * 40)

    for name in sorted(models.keys()):
        model, long_flag = models[name]
        cap = generate(model, img_path, long=long_flag)
        label = name.split("_", 1)[1]
        print(f"[{label}] ({len(cap.split())} words):")
        print(f"  {cap}")
        print()

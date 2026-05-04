import torch, os, json
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    InstructBlipForConditionalGeneration,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)

TEST_DIR = "/home/drive4/figcaps_data/No-Subfig-Img/test"
JSON_DIR = "/home/drive4/figcaps_data/Caption-All/test"

prompt = "Describe this scientific figure in detail. Include the figure type, what the axes represent, the main data trends, and the key conclusion. Write 100-150 words."

# Pick 5 diverse test figures
all_imgs = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])[:100]
selected = all_imgs[::20]  # 5 figures
print(f"Testing on {len(selected)} figures\n")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# ─── Model 1: Qwen2.5-VL-7B ───
print("=" * 80)
print("Loading Qwen2.5-VL-7B...")
qwen_proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="cuda:0"
)

def run_qwen(img_path):
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": prompt}
    ]}]
    text = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = qwen_proc(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=300)
    # Strip input tokens from output
    out_trimmed = out[0][inputs.input_ids.shape[1]:]
    return qwen_proc.decode(out_trimmed, skip_special_tokens=True)

# ─── Model 2: BLIP-2-OPT-2.7B ───
print("Loading BLIP-2-OPT-2.7B...")
blip2_proc = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="cuda:2"
)

def run_blip2(img_path):
    raw_image = Image.open(img_path).convert("RGB")
    inputs = blip2_proc(images=raw_image, text=prompt, return_tensors="pt", padding=True).to("cuda:2")
    with torch.no_grad():
        out = blip2_model.generate(**inputs, max_new_tokens=300)
    return blip2_proc.decode(out[0], skip_special_tokens=True)

# ─── Model 3: InstructBLIP-7B ───
print("Loading InstructBLIP-7B...")
try:
    instruct_proc = AutoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", use_fast=False)
    instruct_model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-7b",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="cuda:1"
    )
    instruct_loaded = True
except Exception as e:
    print(f"  InstructBLIP load error: {e}")
    instruct_loaded = False

def run_instructblip(img_path):
    if not instruct_loaded:
        return "[LOAD FAILED]"
    raw_image = Image.open(img_path).convert("RGB")
    inputs = instruct_proc(images=raw_image, text=prompt, return_tensors="pt", padding=True).to("cuda:1")
    with torch.no_grad():
        out = instruct_model.generate(**inputs, max_new_tokens=300)
    return instruct_proc.decode(out[0], skip_special_tokens=True)

# ─── RUN ALL ───
print("\n" + "=" * 80)
print("STARTING COMPARISON")
print("=" * 80)

for img_name in selected:
    img_path = os.path.join(TEST_DIR, img_name)
    jp = os.path.join(JSON_DIR, img_name.replace(".png", ".json"))
    if not os.path.exists(jp):
        continue
    with open(jp) as f:
        d = json.load(f)
    gt = d.get("0-originally-extracted", "")

    print("\n" + "=" * 80)
    print(f"FIGURE: {img_name}")
    print(f"GT: {gt[:200]}")
    print("-" * 40)

    for name, fn in [("Qwen2.5-VL-7B", run_qwen), ("BLIP-2-2.7B", run_blip2), ("InstructBLIP-7B", run_instructblip)]:
        try:
            result = fn(img_path)
            words = len(result.split())
            print(f"\n[{name}] ({words} words):")
            print(f"  {result}")
        except Exception as e:
            print(f"\n[{name}] ERROR: {e}")

print("\n\n🏁 REGRESSION TEST COMPLETE")

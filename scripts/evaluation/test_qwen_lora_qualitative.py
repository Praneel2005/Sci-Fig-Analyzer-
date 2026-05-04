import torch, os, json
import ssl, requests, urllib3
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# --- SSL Bypass ---
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()
os.environ['CURL_CA_BUNDLE'] = ''
old_req = requests.Session.request
def new_req(*a, **kw):
    kw['verify'] = False
    return old_req(*a, **kw)
requests.Session.request = new_req
# ------------------

LORA_DIR = "/home/drive4/qwen_scicap_lora/checkpoint-9375"
TEST_DIR = "/home/drive4/figcaps_data/No-Subfig-Img/test"
JSON_DIR = "/home/drive4/figcaps_data/Caption-All/test"

print("Loading Base Model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="cuda:0",
)

print(f"Loading LoRA weights from {LORA_DIR}...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

prompt = "Describe this scientific figure in detail. Include the figure type, axes, trends, and key conclusion. Write 100-150 words."

# Pick 5 test figures
all_imgs = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])[:100]
selected = all_imgs[::20]

print("\n" + "="*80)
print("TESTING FINE-TUNED QWEN LORA")
print("="*80)

for img_name in selected:
    img_path = os.path.join(TEST_DIR, img_name)
    jp = os.path.join(JSON_DIR, img_name.replace(".png", ".json"))
    if not os.path.exists(jp): continue
    
    with open(jp) as f: d = json.load(f)
    gt = d.get("0-originally-extracted", "")

    raw_image = Image.open(img_path).convert("RGB")
    
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to("cuda:0")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300)
    
    out_trimmed = out[0][inputs.input_ids.shape[1]:]
    result = processor.decode(out_trimmed, skip_special_tokens=True)

    print(f"\nFIGURE: {img_name}")
    print(f"GT CAPTION: {gt[:100]}...")
    print(f"OUR MODEL ({len(result.split())} words):\n{result}")
    print("-" * 80)

print("Done!")

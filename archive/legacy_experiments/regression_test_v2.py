import torch, os, json
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration, 
    InstructBlipForConditionalGeneration,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

# Use a real image we just extracted
IMG_DIR = "/home/drive4/scicap_plus_raw/test_images_subset"
img_name = os.listdir(IMG_DIR)[0]
img_path = os.path.join(IMG_DIR, img_name)

prompt = "Describe this scientific figure in detail. Include the figure type, axes, trends, and key conclusion. Write 100-150 words."

print(f"🚀 Testing models on image: {img_name}")

results = {}

def test_model(name, model_id, model_class, device):
    print(f"\n--- Testing {name} ---")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = model_class.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map=device,
            load_in_4bit=True
        )
        
        if "Qwen" in name:
            from qwen_vl_utils import process_vision_info
            messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
        else:
            raw_image = Image.open(img_path).convert("RGB")
            inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)

        out = model.generate(**inputs, max_new_tokens=300)
        results[name] = processor.decode(out[0], skip_special_tokens=True)
        print(f"✅ {name} Success.")
    except Exception as e:
        print(f"❌ {name} Failed: {e}")

# Run the 3-way test
test_model("Qwen2.5-VL-7B", "Qwen/Qwen2.5-VL-7B-Instruct", Qwen2_5_VLForConditionalGeneration, "cuda:0")
test_model("InstructBLIP-7B", "Salesforce/instructblip-vicuna-7b", InstructBlipForConditionalGeneration, "cuda:1")
test_model("BLIP-2-2.7B", "Salesforce/blip2-opt-2.7b", Blip2ForConditionalGeneration, "cuda:2")

print("\n" + "="*80)
print("FINAL SCIENTIFIC PARAGRAPH COMPARISON")
print("="*80)
for name, res in results.items():
    print(f"\n[{name}]:\n{res}\n")

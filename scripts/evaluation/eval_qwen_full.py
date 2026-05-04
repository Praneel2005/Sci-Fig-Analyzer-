import os
import json
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
from nltk.translate import meteor_score

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ── Config ────────────────────────────────────────────────────────
TEST_DIR    = "/home/drive4/figcaps_data/No-Subfig-Img/test"
JSON_DIR    = "/home/drive4/figcaps_data/Caption-All/test"
MODEL_NAME  = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_FILE = "/home/drive4/FigCapsHF/qwen_full_test_results.json"
CHECKPOINT_INTERVAL = 100

PROMPT = """You are a scientific figure analyst. Examine this figure carefully and write a detailed paragraph description.

Your description must cover:
1. What type of figure this is
2. What the axes, labels, or variables represent
3. The main trend, pattern, or finding visible in the figure
4. The key conclusion a reader should draw

Important rules:
- Only describe what you can actually see in the figure
- Do not invent experimental details not shown in the image
- Write 100-150 words in clear academic English"""

def load_progress():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(results):
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def evaluate_metrics(results_dict):
    print("\n" + "="*50)
    print("COMPUTING QUANTITATIVE METRICS")
    print("="*50)
    
    references = []
    hypotheses = []
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_r, rouge2_r, rougeL_r = 0, 0, 0
    rouge1_f, rouge2_f, rougeL_f = 0, 0, 0
    meteor_total = 0
    
    for img_name, data in results_dict.items():
        ref = data['ground_truth']
        hyp = data['generated']
        
        references.append([ref.split()])
        hypotheses.append(hyp.split())
        
        # ROUGE
        scores = scorer.score(ref, hyp)
        # Recall is important here because hyp is much longer than ref
        rouge1_r += scores['rouge1'].recall
        rouge2_r += scores['rouge2'].recall
        rougeL_r += scores['rougeL'].recall
        rouge1_f += scores['rouge1'].fmeasure
        rouge2_f += scores['rouge2'].fmeasure
        rougeL_f += scores['rougeL'].fmeasure
        
        # METEOR
        meteor_total += meteor_score.single_meteor_score(ref.split(), hyp.split())
        
    n = len(results_dict)
    if n == 0:
        return
        
    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    print(f"Total Evaluated: {n} images")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"METEOR: {meteor_total/n:.4f}")
    print(f"ROUGE-L (Recall): {rougeL_r/n:.4f}  <-- Very important when comparing paragraphs to short captions!")
    print(f"ROUGE-L (F1):     {rougeL_f/n:.4f}")
    print("="*50)

def main():
    print("Loading Qwen2.5-VL-7B-Instruct...")
    device = "cuda:0"
    
    # Enable tf32 for faster inference on Ada GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("Model loaded.")

    all_imgs = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])
    
    results = load_progress()
    print(f"Found {len(results)} already processed figures.")
    
    to_process = [img for img in all_imgs if img not in results]
    print(f"Figures remaining: {len(to_process)}")

    if not to_process:
        evaluate_metrics(results)
        return

    from qwen_vl_utils import process_vision_info

    for i, img_name in enumerate(tqdm(to_process)):
        img_path = os.path.join(TEST_DIR, img_name)
        jp = os.path.join(JSON_DIR, img_name.replace(".png", ".json"))
        
        if not os.path.exists(jp): continue
        with open(jp) as f: d = json.load(f)
        
        gt = d.get("0-originally-extracted", "")
        
        try:
            image = Image.open(img_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT}
            ]}]
            
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)

            inputs = processor(
                text=[text_input],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

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

            input_len = inputs["input_ids"].shape[1]
            generated = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

            results[img_name] = {
                "ground_truth": gt,
                "generated": generated,
                "word_count": len(generated.split())
            }
            
        except Exception as e:
            print(f"Error on {img_name}: {e}")
            continue
            
        # Save periodically
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_progress(results)

    # Final save and evaluate
    save_progress(results)
    evaluate_metrics(results)

if __name__ == "__main__":
    main()

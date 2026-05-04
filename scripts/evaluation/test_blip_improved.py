import torch
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import BlipForConditionalGeneration, AutoProcessor
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as single_meteor
from nltk.tokenize import word_tokenize
import ssl
import requests
import urllib3

# SSL bypass for restricted server
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()
os.environ['CURL_CA_BUNDLE'] = ''
old_session_request = requests.Session.request
def new_session_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_session_request(*args, **kwargs)
requests.Session.request = new_session_request

# Setup NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def compute_metrics(preds, refs):
    if not preds or not refs:
        return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'bleu1': 0, 'bleu4': 0, 'meteor': 0}
        
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1, r2, rL = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rL.append(s['rougeL'].fmeasure)
    
    refs_tokenized = [[word_tokenize(r.lower())] for r in refs]
    preds_tokenized = [word_tokenize(p.lower()) for p in preds]
    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(1, 0, 0, 0), smoothing_function=smooth)
    b4 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    m_scores = [single_meteor([word_tokenize(r)], word_tokenize(p)) for p, r in zip(preds, refs)]
    
    return {
        'rouge1': sum(r1)/len(r1), 'rouge2': sum(r2)/len(r2), 'rougeL': sum(rL)/len(rL),
        'bleu1': b1, 'bleu4': b4, 'meteor': sum(m_scores)/len(m_scores)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--benchmark_path', type=str, default='/home/drive4/figcaps_data')
    parser.add_argument('--num_samples', type=int, default=500)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model_path}...")
    
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    state_dict = torch.load(args.model_path, map_location='cpu')
    if 'model' in state_dict: state_dict = state_dict['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    test_json_dir = os.path.join(args.benchmark_path, 'Caption-All', 'test')
    img_dir = os.path.join(args.benchmark_path, 'No-Subfig-Img', 'test')
    
    files = sorted([f for f in os.listdir(test_json_dir) if f.endswith('.json')])
    preds, refs = [], []

    print(f"Evaluating on {len(files)} samples...")
    for fname in tqdm(files):
        with open(os.path.join(test_json_dir, fname)) as f:
            data = json.load(f)
        
        img_path = os.path.join(img_dir, fname.replace('.json', '.png'))
        if not os.path.exists(img_path): continue
        
        image = Image.open(img_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Use improved generation parameters
            out = model.generate(
                **inputs, 
                max_length=512, 
                min_length=10, 
                num_beams=5,
                repetition_penalty=1.3
            )
            caption = processor.decode(out[0], skip_special_tokens=True)
            
        preds.append(caption)
        refs.append(data.get('0-originally-extracted', ''))

    results = compute_metrics(preds, refs)
    print("\n" + "="*30 + "\nRESULTS\n" + "="*30)
    for k, v in results.items(): print(f"{k.upper()}: {v:.4f}")
    
    # Save results to a file for summary
    output_name = f"eval_{os.path.basename(os.path.dirname(args.model_path))}.json"
    with open(output_name, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

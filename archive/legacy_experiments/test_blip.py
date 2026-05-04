"""
test_blip.py  —  Full quantitative evaluation on the test set.
Uses rouge_score, nltk directly (no HuggingFace 'evaluate' library)
to avoid SSL/network issues on restricted servers.
"""
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import os
import ssl
import urllib3
import numpy as np
import json
import argparse
import pandas as pd
from tqdm import tqdm

# SSL bypass for HuggingFace tokenizer download
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()
os.environ["CURL_CA_BUNDLE"] = ""
old_session_request = requests.Session.request
def new_session_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_session_request(*args, **kwargs)
requests.Session.request = new_session_request

# Local metric imports (no network needed)
from rouge_score import rouge_scorer
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as single_meteor
from nltk.tokenize import word_tokenize


def compute_metrics(all_predictions, all_references):
    """Compute ROUGE, BLEU, METEOR from accumulated predictions/references."""
    # --- ROUGE ---
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for pred, ref in zip(all_predictions, all_references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    # --- BLEU ---
    # corpus_bleu expects list of [reference_tokens] and list of hypothesis_tokens
    refs_tokenized = [[word_tokenize(ref.lower())] for ref in all_references]
    preds_tokenized = [word_tokenize(pred.lower()) for pred in all_predictions]
    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # --- METEOR ---
    meteor_scores = []
    for pred, ref in zip(all_predictions, all_references):
        meteor_scores.append(single_meteor([word_tokenize(ref)], word_tokenize(pred)))

    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores),
        'bleu1': bleu1,
        'bleu4': bleu4,
        'meteor': np.mean(meteor_scores),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='BLIP test script')
    parser.add_argument('--benchmark_path', help='Path to the benchmark dataset')
    parser.add_argument('--model_path', help='Path to the model')
    args = parser.parse_args()

    def load_demo_image(image_path, device):
        raw_image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image = transform(raw_image).unsqueeze(0).to(device)
        return image

    model_path = args.model_path
    model = blip_decoder(pretrained=model_path, vit='base')
    model.eval()
    model = model.to(device)
    batch_size = 128
    benchmark_path = args.benchmark_path
    test_json_path = os.path.join(benchmark_path, 'No-Subfig-Img', 'test/metadata.jsonl')
    json_df = pd.read_json(test_json_path, lines=True)
    json_df['file_name'] = os.path.join(benchmark_path, "No-Subfig-Img", "test") + "/" + json_df['file_name']
    captions_list = json_df['text'].tolist()
    image_list = json_df['file_name'].tolist()

    print(f"Total test images: {len(captions_list)}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {len(range(0, len(captions_list)-batch_size+1, batch_size))}")

    all_predictions = []
    all_references = []

    for i in tqdm(range(0, len(captions_list)-batch_size+1, batch_size)):
        gt_captions = [captions_list[i+k] for k in range(batch_size)]
        image_paths = [image_list[i+k] for k in range(batch_size)]
        image = torch.cat([load_demo_image(image_path=image_path, device=device) for image_path in image_paths])
        with torch.no_grad():
            caption = model.generate(image, sample=True, top_p=0.7, max_length=512, min_length=10)
            all_predictions.extend(caption)
            all_references.extend(gt_captions)

    print(f"\nTotal samples evaluated: {len(all_predictions)}")
    print("Computing metrics...")

    results = compute_metrics(all_predictions, all_references)

    print("\n" + "=" * 60)
    print("QUANTITATIVE EVALUATION RESULTS")
    print("=" * 60)
    print(f"ROUGE-1:  {results['rouge1']:.4f}")
    print(f"ROUGE-2:  {results['rouge2']:.4f}")
    print(f"ROUGE-L:  {results['rougeL']:.4f}")
    print(f"BLEU-1:   {results['bleu1']:.4f}")
    print(f"BLEU-4:   {results['bleu4']:.4f}")
    print(f"METEOR:   {results['meteor']:.4f}")
    print("=" * 60)

    # Save results to JSON for results_table.py
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to eval_results.json")


if __name__ == '__main__':
    main()

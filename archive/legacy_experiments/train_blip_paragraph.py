"""
train_blip_paragraph.py
Trains BLIP to generate paragraph-length descriptions.
Uses ~/figcaps_data_paragraph as dataset (expanded captions).
Key difference from train_blip_improved.py:
  - max_length increased to 512
  - trains on paragraph-length targets (100-200 words)
  - quality threshold lowered slightly (more data for harder task)
"""

import argparse
import os
import json
import torch
import ssl
import urllib3
import requests
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image

# SSL bypass for restricted server environment
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()
os.environ["CURL_CA_BUNDLE"] = ""
old_session_request = requests.Session.request
def new_session_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_session_request(*args, **kwargs)
requests.Session.request = new_session_request

old_request = requests.request
def new_request(method, url, **kwargs):
    kwargs['verify'] = False
    return old_request(method, url, **kwargs)
requests.request = new_request
requests.get = lambda url, **kwargs: new_request('get', url, **kwargs)

class ParagraphCaptioningDataset(Dataset):

    def __init__(self, img_dir, json_dir, processor,
                 hf_score_type="helpfulness",
                 quality_threshold=0.30,
                 max_ocr_tokens=25):
        self.img_dir        = img_dir
        self.json_dir       = json_dir
        self.processor      = processor
        self.hf_score_type  = hf_score_type
        self.max_ocr_tokens = max_ocr_tokens

        all_names = sorted([
            f.replace(".json", "")
            for f in os.listdir(json_dir)
            if f.endswith(".json")
        ])

        self.file_list = []
        skipped = 0
        for name in all_names:
            img_path  = os.path.join(img_dir,  name + ".png")
            json_path = os.path.join(json_dir, name + ".json")
            if not os.path.exists(img_path):
                continue
            try:
                with open(json_path) as f:
                    data = json.load(f)
                score = data["human-feedback"][hf_score_type]["score"]
                if score < quality_threshold:
                    skipped += 1
                    continue
                self.file_list.append(name)
            except (KeyError, TypeError, json.JSONDecodeError):
                self.file_list.append(name)

        print(f"[ParagraphDataset] {len(self.file_list)} examples "
              f"| {skipped} filtered | threshold={quality_threshold}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name      = self.file_list[idx]
        img_path  = os.path.join(self.img_dir,  name + ".png")
        json_path = os.path.join(self.json_dir, name + ".json")

        image = Image.open(img_path).convert("RGB")

        with open(json_path) as f:
            data = json.load(f)

        # Use the expanded paragraph caption as training target
        caption_prepend = data["human-feedback"][self.hf_score_type]["caption-prepend"]

        # Also inject OCR context
        ocr_tokens = data.get("Img-text", [])
        if ocr_tokens:
            ocr_str    = " ".join(str(t) for t in ocr_tokens[:self.max_ocr_tokens])
            text_input = f"Figure text: {ocr_str.strip()}. {caption_prepend}"
        else:
            text_input = caption_prepend

        # max_length=512 to accommodate paragraph targets
        encoding = self.processor(
            images=image,
            text=text_input,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def training_function(config, args):
    accelerator = Accelerator(
        cpu=args.cpu,
        mixed_precision=args.mixed_precision,
        project_dir=args.logging_dir
    )

    accelerator.print("Loading processor...")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    img_dir  = os.path.join(args.benchmark_path, "No-Subfig-Img", "train") # Adjusted to lowercase
    json_dir = os.path.join(args.benchmark_path, "Caption-All",   "train") # Adjusted to lowercase

    train_dataset = ParagraphCaptioningDataset(
        img_dir           = img_dir,
        json_dir          = json_dir,
        processor         = processor,
        hf_score_type     = args.hf_score_type,
        quality_threshold = 0.30,
        max_ocr_tokens    = 25
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle    = True,
        batch_size = config["batch_size"],
        num_workers= 4,
        pin_memory = True
    )

    # Load model — optionally start from improved checkpoint
    if args.start_from_checkpoint:
        accelerator.print(f"Loading from checkpoint: {args.start_from_checkpoint}")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        state = torch.load(args.start_from_checkpoint, map_location="cpu")
        # Handle both raw state dict and accelerate-saved formats
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    else:
        accelerator.print("Loading base BLIP model from HuggingFace...")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    model = model.to(accelerator.device)
    accelerator.print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    os.makedirs(args.output_dir, exist_ok=True)

    start_epoch = 0
    if args.resume_from_checkpoint and 'epoch_' in args.resume_from_checkpoint:
        start_epoch = int(args.resume_from_checkpoint.split('epoch_')[-1])
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        total_loss  = 0.0
        num_samples = 0

        loop = tqdm(
            enumerate(train_dataloader),
            total   = len(train_dataloader),
            disable = not accelerator.is_local_main_process,
            desc    = f"Epoch {epoch+1}/{config['num_epochs']}"
        )

        for idx, batch in loop:
            input_ids    = batch.pop("input_ids").to(accelerator.device)
            pixel_values = batch.pop("pixel_values").to(accelerator.device)

            outputs = model(
                input_ids    = input_ids,
                pixel_values = pixel_values,
                labels       = input_ids
            )
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                accelerator.print(f"NaN loss at epoch {epoch+1} step {idx} — skipping.")
                optimizer.zero_grad()
                continue

            total_loss  += loss.detach().float()
            num_samples += input_ids.shape[0]
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if num_samples > 0:
                loop.set_postfix(loss=f"{total_loss.item()/num_samples:.4f}")

        avg = total_loss.item() / max(num_samples, 1)
        accelerator.print(f"Epoch {epoch+1} done | avg_loss={avg:.4f}")

        out = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        accelerator.save_state(out)
        accelerator.print(f"Saved: {out}")

    accelerator.print("Paragraph training complete.")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--hf_score_type', type=str, default='helpfulness')
    parser.add_argument('--benchmark_path', type=str, default='/home/drive4/figcaps_data')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--logging_dir', type=str, default='logs')
    parser.add_argument('--start_from_checkpoint', type=str, default=None)
    args = parser.parse_args()

    config = {
        'lr': 3e-5,
        'num_epochs': 10,
        'seed': 42,
        'batch_size': 4
    }
    
    training_function(config, args)

if __name__ == '__main__':
    main()

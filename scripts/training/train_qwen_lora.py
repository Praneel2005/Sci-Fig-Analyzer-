import torch, json, os, random
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# ─── Dataset ───
class SciCapDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path) as f:
            self.data = json.load(f)
        self.processor = processor
        self.prompt = "Describe this scientific figure in detail. Include the figure type, axes, trends, and key conclusion. Write 100-150 words."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(item["image"]).convert("RGB")
        except:
            return self.__getitem__(random.randint(0, len(self.data)-1))

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": item["paragraph"]}
            ]}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        
        # Explicit resolution constraints for Qwen2.5-VL
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            min_pixels=256*28*28,
            max_pixels=1280*28*28,
            padding=False,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        
        # Mask prompt in labels
        prompt_ids = self.processor.tokenizer.encode("assistant\n", add_special_tokens=False)
        for i in range(len(input_ids) - len(prompt_ids)):
            if input_ids[i:i+len(prompt_ids)].tolist() == prompt_ids:
                labels[:i+len(prompt_ids)] = -100
                break

        res = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"]
        }
        return res

# ─── Training Setup ───
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="cuda:0", # Forces single GPU
)

model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads() # Fixes the gradient warning

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

train_ds = SciCapDataset("/home/drive4/scicap_plus_processed/train.json", processor)
val_ds = SciCapDataset("/home/drive4/scicap_plus_processed/val.json", processor)

def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), processor.tokenizer.pad_token_id)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100)]))
        
    return {
        "input_ids": torch.stack(input_ids).long(),
        "attention_mask": torch.stack(attention_mask).long(),
        "labels": torch.stack(labels).long(),
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0)
    }

training_args = TrainingArguments(
    output_dir="/home/drive4/qwen_scicap_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    fp16=True,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn
)

print("🚀 Starting Training (GPU 0)...")
trainer.train()

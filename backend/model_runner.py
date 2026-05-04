import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

class QwenModelRunner:
    def __init__(self):
        self.device = "cuda:0"
        self.model = None
        self.processor = None
        self.prompt = "You are a scientific figure analyst. Examine this figure carefully and write a detailed paragraph description.\n\nYour description must cover:\n1. What type of figure this is (graph, bar chart, diagram, etc.)\n2. What the axes, labels, or variables represent\n3. The main trend, pattern, or finding visible in the figure\n4. The key conclusion a reader should draw from this figure\n\nImportant rules:\n- Only describe what you can actually see in the figure\n- Do not invent experimental details not shown in the image\n- Write 100-150 words in clear academic English\n- Do not use bullet points — write flowing paragraph text"

    def load_model(self):
        if self.model is None:
            print("Loading Qwen2.5-VL-7B-Instruct into memory...")
            torch.backends.cuda.matmul.allow_tf32 = True
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
            print("Model loaded successfully.")

    def generate_paragraph(self, image_path):
        self.load_model()
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt}
                ]
            }
        ]

        from qwen_vl_utils import process_vision_info
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=250,
                min_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4
            )

        input_len = inputs["input_ids"].shape[1]
        generated = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        return generated

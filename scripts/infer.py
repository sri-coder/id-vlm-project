"""
Run the fine-tuned model on a single image -- useful for demos, README
screenshots, or quick sanity checks after training.

Usage:
    python infer.py --checkpoint outputs/final --image path/to/id.jpg
"""
import argparse
import json

import torch
from PIL import Image
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
TARGET_FIELDS = ["name", "surname", "date_of_birth", "document_number", "expiry_date"]
PROMPT = (
    "Extract the following fields from this identity document image and "
    "return ONLY a JSON object with these keys: "
    f"{', '.join(TARGET_FIELDS)}. "
    "If a field is not visible or not present, use null."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    image = Image.open(args.image).convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200)
    output_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]

    print("Raw output:", output_text)
    try:
        print("Parsed:", json.dumps(json.loads(output_text.strip()), indent=2))
    except json.JSONDecodeError:
        print("(output was not valid JSON)")


if __name__ == "__main__":
    main()

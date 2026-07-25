"""
Local smoke test -- run this on your RTX 3050 laptop BEFORE touching Colab.

Purpose: confirm the model loads in 4-bit, an image loads, generation runs,
and the output is parseable. This catches 90% of bugs (wrong image format,
bad prompt template, missing package) for free, before burning your Colab
GPU quota debugging the same issues.

Usage:
    python local_smoke_test.py --image path/to/some_id_image.jpg
"""
import argparse
import json

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image

TARGET_FIELDS = ["name", "surname", "date_of_birth", "document_number", "expiry_date"]

PROMPT = (
    "Extract the following fields from this identity document image and "
    "return ONLY a JSON object with these keys: "
    f"{', '.join(TARGET_FIELDS)}. "
    "If a field is not visible or not present, use null."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct"
    )
    args = parser.parse_args()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name)

    print("Loading model in 4-bit (this should fit comfortably on a 4-6GB GPU)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(
        model.device
    )

    print("Running generation...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200)

    output_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]

    print("\n--- Raw model output ---")
    print(output_text)

    print("\n--- Parse check ---")
    try:
        parsed = json.loads(output_text.strip())
        print("Valid JSON. Parsed fields:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print(
            "Output was not valid JSON. This is expected before fine-tuning -- "
            "the base model hasn't learned your output format yet. What matters "
            "right now is that the pipeline ran end-to-end without crashing."
        )

    print("\nSmoke test complete. If you got here without an error, your local "
          "environment is good and you're ready to move training to Colab.")


if __name__ == "__main__":
    main()

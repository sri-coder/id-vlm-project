import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
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


# ---- Corruption functions ---------------------------------------------
# Each takes a PIL image and returns a corrupted PIL image. Keep these
# simple and inspectable -- you want to be able to explain exactly what
# each one does in your writeup.

def corrupt_blur(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    blurred = cv2.GaussianBlur(arr, (9, 9), sigmaX=4)
    return Image.fromarray(blurred)


def corrupt_rotate(img: Image.Image) -> Image.Image:
    return img.rotate(12, expand=True, fillcolor=(255, 255, 255))


def corrupt_glare(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    overlay = np.zeros_like(arr)
    cv2.ellipse(
        overlay, (w // 3, h // 3), (w // 4, h // 6), 30, 0, 360, (255, 255, 255), -1
    )
    blended = cv2.addWeighted(arr, 1.0, overlay, 0.5, 0)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def corrupt_occlusion(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    h, w = arr.shape[:2]
    # Block out a random-ish region (fixed here for reproducibility --
    # randomize with a seed if you want variety across the val set)
    y0, y1 = int(h * 0.4), int(h * 0.55)
    x0, x1 = int(w * 0.1), int(w * 0.5)
    arr[y0:y1, x0:x1] = 0
    return Image.fromarray(arr)


def corrupt_jpeg(img: Image.Image, quality: int = 15) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


CORRUPTIONS = {
    "clean": lambda img: img,
    "blur": corrupt_blur,
    "rotate": corrupt_rotate,
    "glare": corrupt_glare,
    "occlusion": corrupt_occlusion,
    "jpeg_compression": corrupt_jpeg,
}


# ---- Model wrapper -------------------------------------------------------

def load_model(checkpoint_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    return model, processor


def run_inference(model, processor, image: Image.Image) -> dict:
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
    try:
        return json.loads(output_text.strip())
    except json.JSONDecodeError:
        return {field: None for field in TARGET_FIELDS}


def field_accuracy(pred: dict, gold: dict) -> float:
    """Exact-match accuracy averaged across fields present in gold."""
    correct, total = 0, 0
    for field in TARGET_FIELDS:
        gold_val = gold.get(field)
        if gold_val is None:
            continue
        total += 1
        pred_val = pred.get(field)
        if pred_val is not None and str(pred_val).strip().lower() == str(gold_val).strip().lower():
            correct += 1
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    print("Loading model...")
    model, processor = load_model(args.checkpoint)

    records = []
    with open(args.val_file, "r") as f:
        for line in f:
            records.append(json.loads(line))
            if args.max_samples and len(records) >= args.max_samples:
                break

    results = {name: [] for name in CORRUPTIONS}

    for i, rec in enumerate(records):
        print(f"Evaluating sample {i + 1}/{len(records)}...")
        image = Image.open(rec["image"]).convert("RGB")
        gold = rec["fields"]

        for corruption_name, corrupt_fn in CORRUPTIONS.items():
            corrupted_image = corrupt_fn(image)
            pred = run_inference(model, processor, corrupted_image)
            acc = field_accuracy(pred, gold)
            results[corruption_name].append(acc)

    print("\n=== Adversarial Evaluation Results ===")
    print(f"{'Corruption':<20}{'Mean Field Accuracy':<20}{'N Samples'}")
    summary = {}
    for corruption_name, accs in results.items():
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        summary[corruption_name] = mean_acc
        print(f"{corruption_name:<20}{mean_acc:<20.3f}{len(accs)}")

    out_path = Path(args.checkpoint) / "adversarial_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()

"""
Prepare training data from CORD-v2 (naver-clova-ix/cord-v2), a public
receipt-understanding dataset on Hugging Face with image + structured field
annotations already included. Using this instead of MIDV skips the manual
download/unzip/parse step entirely -- the dataset loads directly.

CORD-v2 fields are receipt-style (store name, item names, prices, total)
rather than ID-document-style, but the *task* is identical to what
HyperVerge cares about: extracting structured fields from a photographed
document under real-world visual noise. That's the point you make in your
README -- the method transfers, and you can note ID documents as the
natural next dataset.

Usage:
    python prepare_cord.py --out_dir data/processed --max_train 500 --max_val 100
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from PIL import Image

# The fields we'll train the model to extract. CORD-v2 ground truth is
# nested JSON (menu items, subtotal, total, etc.) -- we flatten to a
# manageable flat set for a first pass. Extend this once you've looked at
# a few real examples.
TARGET_FIELDS = ["store_name", "total_price", "date", "item_names"]

INSTRUCTION = (
    "Extract the following fields from this receipt image and return ONLY "
    f"a JSON object with these keys: {', '.join(TARGET_FIELDS)}. "
    "If a field is not visible or not present, use null."
)


def flatten_to_str(value):
    """CORD-v2's schema is inconsistent across examples -- the same logical
    field (e.g. an item name) can appear as a plain string in one record and
    as a list of strings in another (when a receipt has duplicate/repeated
    entries). This normalizes any value down to a single string so the rest
    of the pipeline never has to special-case types."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [flatten_to_str(v) for v in value]
        parts = [p for p in parts if p]
        return ", ".join(parts) if parts else None
    if isinstance(value, dict):
        # some fields nest one level deeper (e.g. {"nm": "..."})
        return flatten_to_str(value.get("nm") or value.get("text") or list(value.values()))
    return str(value)


def extract_fields(ground_truth_str: str) -> dict:
    """CORD-v2 ground truth is a JSON string under a 'gt_parse' key with
    nested structure. This pulls out a flat set of fields -- it's
    intentionally simple; refine once you've inspected real samples."""
    try:
        gt = json.loads(ground_truth_str)
    except json.JSONDecodeError:
        return {field: None for field in TARGET_FIELDS}

    parse = gt.get("gt_parse", gt)

    menu = parse.get("menu")

    store_name = None
    if isinstance(menu, dict):
        store_name = flatten_to_str(menu.get("nm"))
    elif isinstance(menu, list) and menu:
        store_name = flatten_to_str(menu[0].get("nm")) if isinstance(menu[0], dict) else None

    total_block = parse.get("total", {})
    total_price = flatten_to_str(total_block.get("total_price")) if isinstance(total_block, dict) else None

    item_names = None
    if isinstance(menu, list):
        names = [flatten_to_str(item.get("nm")) for item in menu if isinstance(item, dict)]
        names = [n for n in names if n]
        item_names = ", ".join(names) if names else None

    sub_total = parse.get("sub_total", {})
    date = flatten_to_str(sub_total.get("date")) if isinstance(sub_total, dict) else None

    return {
        "store_name": store_name,
        "total_price": total_price,
        "date": date,
        "item_names": item_names,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_train", type=int, default=500)
    parser.add_argument("--max_val", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CORD-v2 from Hugging Face (this happens once, cached after)...")
    dataset = load_dataset("naver-clova-ix/cord-v2")

    def process_split(split_name, max_samples, out_filename):
        split = dataset[split_name]
        n = min(max_samples, len(split)) if max_samples else len(split)
        out_path = out_dir / out_filename
        written = 0
        with open(out_path, "w") as f:
            for i in range(n):
                example = split[i]
                image: Image.Image = example["image"]
                fields = extract_fields(example["ground_truth"])

                img_path = images_dir / f"{split_name}_{i}.jpg"
                image.convert("RGB").save(img_path, "JPEG")

                record = {
                    "image": str(img_path),
                    "instruction": INSTRUCTION,
                    "fields": fields,
                }
                f.write(json.dumps(record) + "\n")
                written += 1
        print(f"Wrote {written} records to {out_path}")

    process_split("train", args.max_train, "train.jsonl")
    process_split("validation", args.max_val, "val.jsonl")


if __name__ == "__main__":
    main()

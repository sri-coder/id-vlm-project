
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


def extract_fields(ground_truth_str: str) -> dict:
    """CORD-v2 ground truth is a JSON string under a 'gt_parse' key with
    nested structure. This pulls out a flat set of fields -- it's
    intentionally simple; refine once you've inspected real samples."""
    try:
        gt = json.loads(ground_truth_str)
    except json.JSONDecodeError:
        return {field: None for field in TARGET_FIELDS}

    parse = gt.get("gt_parse", gt)

    store_name = None
    if isinstance(parse.get("menu"), dict):
        store_name = parse.get("menu", {}).get("nm")

    total_price = None
    total_block = parse.get("total", {})
    if isinstance(total_block, dict):
        total_price = total_block.get("total_price")

    item_names = None
    menu = parse.get("menu")
    if isinstance(menu, list):
        names = [item.get("nm") for item in menu if isinstance(item, dict) and item.get("nm")]
        item_names = ", ".join(names) if names else None

    date = None
    sub_total = parse.get("sub_total", {})
    if isinstance(sub_total, dict):
        date = sub_total.get("date")

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

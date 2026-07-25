import argparse
import json
import random
from pathlib import Path

# Fields we want the model to learn to extract. Adjust to match what's
# actually present in the MIDV ground truth files you download -- field
# names vary slightly by document type.
TARGET_FIELDS = ["name", "surname", "date_of_birth", "document_number", "expiry_date"]

INSTRUCTION = (
    "Extract the following fields from this identity document image and "
    "return ONLY a JSON object with these keys: "
    f"{', '.join(TARGET_FIELDS)}. "
    "If a field is not visible or not present, use null."
)


def find_pairs(raw_dir: Path):
    """Yield (image_path, annotation_dict) pairs found under raw_dir."""
    pairs = []
    for gt_file in raw_dir.rglob("*.json"):
        img_candidates = list(gt_file.parent.parent.glob(f"images/{gt_file.stem}.*"))
        if not img_candidates:
            continue
        try:
            with open(gt_file, "r") as f:
                ann = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        pairs.append((img_candidates[0], ann))
    return pairs


def normalize_fields(ann: dict) -> dict:
    """Map whatever keys MIDV uses onto our TARGET_FIELDS. Extend this
    mapping once you've inspected a few real annotation files -- MIDV's
    schema varies by document type, so this is intentionally a starting
    point, not a finished mapping."""
    out = {}
    lower_ann = {k.lower(): v for k, v in ann.items()}
    for field in TARGET_FIELDS:
        out[field] = lower_ann.get(field, None)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(raw_dir)
    if not pairs:
        print(
            f"No (image, annotation) pairs found under {raw_dir}. "
            "Check that MIDV data is unzipped there and matches the expected "
            "images/ + ground_truth/ layout -- adjust find_pairs() if your "
            "download has a different structure."
        )
        return

    random.seed(args.seed)
    random.shuffle(pairs)

    n_val = max(1, int(len(pairs) * args.val_split))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    def write_split(split_pairs, filename):
        path = out_dir / filename
        with open(path, "w") as f:
            for img_path, ann in split_pairs:
                fields = normalize_fields(ann)
                record = {
                    "image": str(img_path),
                    "instruction": INSTRUCTION,
                    "fields": fields,
                }
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(split_pairs)} records to {path}")

    write_split(train_pairs, "train.jsonl")
    write_split(val_pairs, "val.jsonl")


if __name__ == "__main__":
    main()

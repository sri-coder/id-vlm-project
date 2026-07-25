"""
LoRA fine-tuning of Qwen2-VL-2B-Instruct -- written as a manual PyTorch
training loop (no transformers.Trainer). This is deliberate: the point of
this project is to demonstrate you understand and control the training
process yourself -- forward pass, loss, backward pass, gradient
accumulation, optimizer step, LR scheduling, checkpointing -- not that you
can call a high-level wrapper.

Run this on Colab/Kaggle (free T4, 16GB). Do not run full training on a
4-6GB laptop GPU -- use local_smoke_test.py locally instead.

Usage:
    python train_lora.py --config ../configs/lora_config.yaml
"""
import argparse
import json
import math
import os

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm import tqdm
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DocFieldDataset(Dataset):
    """Loads (image, instruction, target JSON) records from a JSONL file and
    tokenizes them on the fly. Kept intentionally simple/readable over
    maximally efficient -- readability matters more for a portfolio project
    an interviewer will actually read."""

    def __init__(self, jsonl_path, processor):
        self.records = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.records.append(json.loads(line))
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        target_json = json.dumps(rec["fields"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": rec["instruction"]},
                ],
            },
            {"role": "assistant", "content": target_json},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        image = Image.open(rec["image"]).convert("RGB")

        model_inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}

        # Causal LM: labels are the same sequence as input_ids. We are NOT
        # masking the prompt tokens out of the loss in this first pass --
        # that's a documented next-step improvement (see README), not an
        # oversight. Masking the prompt so loss only counts on the JSON
        # answer typically improves convergence and is a natural
        # "what I'd improve" talking point for an interview.
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs


def collate_fn(batch, pad_token_id):
    """Pads a batch of variable-length tokenized examples. Written by hand
    (rather than relying on a library default collator) so padding and
    attention-mask behavior is fully visible and explainable."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    def pad_1d(tensor, pad_value):
        pad_len = max_len - tensor.shape[0]
        if pad_len == 0:
            return tensor
        pad = torch.full((pad_len,), pad_value, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=0)

    input_ids = torch.stack([pad_1d(b["input_ids"], pad_token_id) for b in batch])
    attention_mask = torch.stack([pad_1d(b["attention_mask"], 0) for b in batch])
    labels = torch.stack([pad_1d(b["labels"], -100) for b in batch])  # -100 = ignored by loss

    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Image-related tensors (pixel_values, image_grid_thw) are fixed-size
    # per image for Qwen2-VL, so they stack directly without padding.
    for key in batch[0].keys():
        if key not in out:
            out[key] = torch.stack([b[key] for b in batch])
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {cfg['model_name']}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["training"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg["model_name"], quantization_config=bnb_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(cfg["model_name"])

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # sanity check: should be a small % of total params
    model.train()

    # ---- Data ----
    train_dataset = DocFieldDataset(cfg["data"]["train_file"], processor)
    val_dataset = DocFieldDataset(cfg["data"]["val_file"], processor)

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["per_device_train_batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # ---- Optimizer + LR schedule ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"],
    )

    grad_accum_steps = cfg["training"]["gradient_accumulation_steps"]
    num_epochs = cfg["training"]["num_train_epochs"]
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    output_dir = cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ---- Manual training loop ----
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        running_loss = 0.0
        optimizer.zero_grad()

        progress = tqdm(train_loader, desc="train")
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps  # normalize for accumulation
            loss.backward()

            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg["training"]["logging_steps"] == 0:
                    avg_loss = running_loss / (step + 1)
                    progress.set_postfix(loss=avg_loss, lr=scheduler.get_last_lr()[0])

        # ---- Validation pass at the end of each epoch ----
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss_total += outputs.loss.item()
        val_loss = val_loss_total / max(1, len(val_loader))
        print(f"Epoch {epoch + 1} val_loss: {val_loss:.4f}")
        model.train()

        # ---- Checkpoint ----
        epoch_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch + 1}")
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = os.path.join(output_dir, "checkpoint-best")
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"New best checkpoint saved (val_loss={val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(output_dir, 'checkpoint-best')}")


if __name__ == "__main__":
    main()

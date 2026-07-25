# Adversarial-Robust Document Field Extraction

Fine-tuning Qwen2-VL-2B with LoRA to extract structured fields from document 
images, and measuring how accuracy holds up under real-world corruptions 
(blur, rotation, glare, occlusion).

## Status
In progress - data pipeline working, training in progress.

## What I'm doing
- Fine-tuning with LoRA using a hand-written PyTorch training loop (no 
  Trainer) so I fully control the forward/backward pass and optimizer step
- Using CORD-v2 (receipts) as a stand-in for ID documents -- same task, 
  public dataset, no manual data collection needed
- Building an eval harness to test robustness under corrupted images

## Why
Experimenting on training and fine tuning models.

## Setup
See `requirements.txt`. Run `data/prepare_cord.py` to get data, 
`scripts/local_smoke_test.py` to sanity-check locally, `scripts/train_lora.py` 
to fine-tune.
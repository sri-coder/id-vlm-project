
# Adversarial-Robust Document Field Extraction (VLM Fine-Tuning)

Fine-tuning Qwen2-VL-2B with LoRA to extract structured fields from
document images, and measuring how accuracy holds up under real-world corruptions.

## Status

Fine-tuning and adversarial evaluation complete. Documented below.

## What I did

- Fine-tuned Qwen2-VL-2B-Instruct with LoRA (rank 16, ~0.83% of
  parameters trainable) using PyTorch training loop --
  no `transformers.Trainer` -- so the forward pass, backward pass,
  gradient accumulation, LR scheduling, and checkpointing are all
  explicit in `scripts/train_lora.py`.
- Used CORD-v2 (public receipt dataset) as a stand-in for ID documents:
  same underlying task (structured field extraction from a photographed
  document under real-world visual noise), public data, no manual
  collection needed.
- Trained the model to output `store_name`, `total_price`, `date`, and
  `item_names` as JSON from a receipt image.
- Built an adversarial evaluation harness (`scripts/corrupt_eval.py`)
  that tests the model against blur, rotation, glare, occlusion, and
  JPEG compression, not just clean images.

## What I found: a real failure, diagnosed and fixed

My first training run plateaued almost immediately -- loss dropped from
8.41 to 6.91 in epoch 1, then stayed flat at 6.91 for two more epochs.
Validation loss barely moved (6.9501 -> 6.9481 -> 6.9474).

**Diagnosis:** my training loop was computing loss over the *entire*
token sequence -- the image tokens and instruction text, not just the
target JSON. Since the image and instruction are identical in structure
across every example, they dominated the loss and diluted the gradient
signal for the one thing that actually mattered: predicting the correct
JSON.

**Fix:** masked the prompt tokens out of the loss (set their label to
`-100`, which PyTorch's loss function ignores), so gradients only flow
from the JSON answer tokens.

**Result after the fix:**

| Epoch | Train loss | Val loss |
|-------|-----------|----------|
| 1 | 0.110 | 0.0584 |
| 2 | 0.0387 | 0.0488 |
| 3 | 0.0189 | 0.0479 |

Train and val loss dropped together (not diverging), which is a good
sign the model is learning the actual task rather than overfitting.

## Adversarial evaluation results

Tested on 20 validation samples, field-level exact-match accuracy:

| Corruption | Mean Field Accuracy |
|---|---|
| Clean | 0.892 |
| Glare | 0.875 |
| JPEG compression | 0.825 |
| Rotation | 0.658 |
| Blur | 0.650 |
| Occlusion | 0.467 |

**Interpretation:** the model is fairly robust to lighting distortion
(glare) and compression artifacts, since these add noise without
destroying character shapes. Blur and rotation cause a meaningful drop,
since they directly distort the spatial/shape information the model
reads text from. Occlusion causes the largest drop by far -- which makes
sense, since blocking part of the image is genuine information loss
(the store name or price may simply not be visible anymore), not just
noise the model can read through. In a production setting, occlusion
would be the priority failure mode to address, likely via multi-frame
capture or explicit "field not visible" confidence signaling rather than
trying to make a single model robust to missing information.

## Known limitations

- `date` field is null in most predictions -- CORD-v2's ground truth
  schema places date information inconsistently across examples, and my
  current field-extraction logic (`data/prepare_cord.py`) doesn't fully
  handle every variant. Worth revisiting with a more thorough schema
  audit.
- CORD-v2 is receipts, not ID documents. The method transfers directly,
  but final accuracy on real ID documents (MIDV-500/2020) would need to
  be validated separately -- `data/prepare_dataset.py` is kept in the
  repo for this next step.
- Adversarial eval was run on 20 samples for time; a larger sample would
  give tighter accuracy estimates.

## What I'd improve with more time

- Fix the date-field extraction schema handling
- Run the full adversarial eval on the complete validation set
- Try higher LoRA rank or more epochs to see if occlusion robustness
  specifically can be improved with targeted data augmentation (training
  on synthetically occluded images)
- Validate on real ID document data (MIDV-500)

## Project structure

d-vlm-project/
├── data/
│ ├── prepare_cord.py # CORD-v2 dataset prep (used for this run)
│ ├── prepare_dataset.py # MIDV-500 prep (next step, not yet run)
│ └── processed/ # generated train/val JSONL + images
├── scripts/
│ ├── local_smoke_test.py # verify local GPU pipeline works
│ ├── train_lora.py # LoRA fine-tuning ( PyTorch loop)
│ ├── corrupt_eval.py # adversarial evaluation harness
│ └── infer.py # run the fine-tuned model on a single image
├── outputs/
│ └── checkpoint-best/ # fine-tuned LoRA adapter (val_loss 0.0479)
├── configs/
│ └── lora_config.yaml
└── requirements.txt




## Setup
See `requirements.txt`. Run `data/prepare_cord.py` to get data, 
`scripts/local_smoke_test.py` to sanity-check locally, `scripts/train_lora.py` 
to fine-tune.
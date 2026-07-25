# Adversarial-Robust ID Document Field Extraction (VLM Fine-Tuning)

Fine-tune a small open Vision-Language Model (Qwen2-VL-2B-Instruct) to extract
structured fields (name, DOB, document number, etc.) from ID document images,
then measure and improve robustness under real-world corruptions (blur, glare,
rotation, occlusion, compression).

This mirrors HyperVerge's actual problem space: document intelligence +
identity verification under messy, adversarial, real-world conditions.

## Project structure

```
id-vlm-project/
├── data/
│   ├── prepare_dataset.py     # turns MIDV-500/2020 into instruction-tuning JSONL
│   └── raw/                   # put downloaded MIDV data here
├── scripts/
│   ├── local_smoke_test.py    # verify your local GPU pipeline works (RTX 3050 friendly)
│   ├── train_lora.py          # LoRA fine-tuning (run on Colab/Kaggle free GPU)
│   ├── corrupt_eval.py        # adversarial evaluation harness
│   └── infer.py               # run the fine-tuned model on a single image
├── configs/
│   └── lora_config.yaml       # LoRA hyperparameters, edit here not in code
├── requirements.txt
└── README.md
```

## Why this hardware split works

Your GeForce RTX 3050 laptop GPU (4–6GB VRAM) is enough to:
- Load Qwen2-VL-2B in 4-bit quantization and confirm your code runs end-to-end
- Debug data loading, prompt formatting, and generation locally, fast, for free

It is **not** meant to run the full LoRA training — do that on Colab (free T4,
16GB) or Kaggle (free T4 x2, 16GB each, 30hrs/week quota). This is completely
normal practice; nobody expects you to own an A100.

## Step-by-step

### 1. Environment setup

```bash
pip install -r requirements.txt
```

### 2. Get the data

Using **CORD-v2** (`naver-clova-ix/cord-v2` on Hugging Face) — a public
receipt-understanding dataset with images and structured field labels
already included, so there's no manual download/unzip step:

```bash
python data/prepare_cord.py --out_dir data/processed --max_train 500 --max_val 100
```

This downloads the dataset (cached after first run), saves images locally,
and produces `data/processed/train.jsonl` / `val.jsonl`, where each line is:
```json
{"image": "path/to/img.jpg", "instruction": "...", "fields": {"store_name": "...", "total_price": "...", "date": "...", "item_names": "..."}}
```

Note: CORD-v2 is receipts, not ID documents — the task (structured field
extraction from a photographed document under real-world visual noise) is
identical to HyperVerge's problem, which is the point made in the writeup.
`data/prepare_dataset.py` (MIDV-500 version) is kept in the repo as the
natural next step once there's time to source ID-document data.

### 3. Local smoke test (run this on your 3050 first)

```bash
python scripts/local_smoke_test.py --image data/processed/sample.jpg
```

This just confirms: model loads in 4-bit, image loads, generation runs,
output is parseable JSON. If this works locally, you know your Colab run
won't fail on silly bugs.

### 4. Fine-tune with LoRA (on Colab/Kaggle GPU)

Upload the repo, then:

```bash
python scripts/train_lora.py --config configs/lora_config.yaml
```

This is a **hand-written PyTorch training loop** — no `transformers.Trainer`.
Forward pass, gradient accumulation, optimizer step, LR scheduling, and
checkpointing are all explicit and readable in `train_lora.py`. This is
intentional: it's the difference between "I called a fine-tuning API" and
"I understand and control the training process," which is exactly what
HyperVerge's JD screens for.

Checkpoints save to `outputs/checkpoint-epoch*` each epoch, and the best
val-loss checkpoint is separately saved to `outputs/checkpoint-best`.

### 5. Adversarial evaluation

```bash
python scripts/corrupt_eval.py --checkpoint outputs/checkpoint-best --val_file data/processed/val.jsonl
```

This reports field-extraction accuracy separately for: clean, blurred,
rotated, glare, occluded, and JPEG-compressed versions of the same images.
This is the table that becomes the centerpiece of your writeup.

### 6. Iterate

Look at which corruption hurts accuracy most. Form a hypothesis (e.g. "blur
destroys thin characters in doc numbers"). Try a fix — usually adding that
corruption type into training data augmentation — and re-run step 4 and 5.
Document the before/after. This loop *is* the project; the first fine-tune
is just the starting point.

## What to put in your final writeup

1. Problem statement (1 paragraph)
2. Data + method (short)
3. Baseline results (clean data accuracy)
4. Adversarial results table (per-corruption accuracy)
5. Your hypothesis → experiment → result iteration (this is the part that
   matters most — most applicants won't have this)
6. What didn't work, and why you think that is
7. Link to GitHub repo

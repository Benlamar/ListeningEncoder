# Real-Time Duplex AI Voice Engine: Setup & Execution Guide

This repository implements the **Two-Tier Full-Duplex Conversational Engine**:
* **Tier 1 (Fast Gatekeeper):** `answerdotai/ModernBERT-base` (or `jhu-clsp/ettin-encoder-150m`), classifying streaming text slices every 600 ms into `LISTEN`, `SPEAK`, `FILLER`, or `INTERRUPT`.
* **Tier 2 (Speaker):** `google/gemma-2-9b-it` or `gemma-3-12b-it` served on vLLM.

---

## 1. Quick Start with `uv` (on your GPU / DGX machine)

This project is configured for **`uv`** via [`pyproject.toml`](pyproject.toml).

```bash
# 1. Create a high-performance virtual environment
uv venv --python 3.11

# 2. Sync all dependencies (PyTorch with CUDA, Transformers, Datasets, Accelerate, PEFT)
uv sync

# 3. (Optional) If you also want vLLM serving on the same machine:
uv sync --extra serving
```

---

## 2. Dataset Preparation

### Option A: Local Micro-Turn Dataset (404,000 Ticks)
The repository contains 539 simulated duplex calls cut into 600 ms ticks under [`models/microturn-lora/dataset/`](models/microturn-lora/dataset/).

To process the full dataset into ModernBERT format:
```bash
uv run python prepare_gatekeeper_dataset.py
```
*Outputs:*
* `data/gatekeeper_train.jsonl` (~404,000 labeled ticks)
* `data/gatekeeper_eval.jsonl` (28 held-out evaluation calls)

To generate a fast smoke-test set (~13,000 rows):
```bash
uv run python prepare_gatekeeper_dataset.py --smoke-test
```

### Option B: Download External Hugging Face Turn-Taking Datasets
To inspect or augment with open benchmarks (`ASLP-lab/Easy-Turn-Testset`, `besimple-ai/duplex-cue`):
```bash
uv run python download_hf_datasets.py
```
*Outputs to:* `data/hf_downloads/`

---

## 3. Training the Gatekeeper (`train_gatekeeper.py`)

The training script automatically calculates **Class-Weighted Cross-Entropy loss** to counteract the 94% `LISTEN` class imbalance, ensuring high recall on rare `INTERRUPT` and `SPEAK` triggers.

### Train ModernBERT-base:
```bash
uv run python train_gatekeeper.py \
  --model_name answerdotai/ModernBERT-base \
  --train_data data/gatekeeper_train.jsonl \
  --eval_data data/gatekeeper_eval.jsonl \
  --output_dir checkpoints/modernbert_gatekeeper \
  --epochs 3 \
  --batch_size 64 \
  --lr 3e-5
```

### Train Ettin-Encoder-150M (Ablation Comparison):
```bash
uv run python train_gatekeeper.py \
  --model_name jhu-clsp/ettin-encoder-150m \
  --train_data data/gatekeeper_train.jsonl \
  --eval_data data/gatekeeper_eval.jsonl \
  --output_dir checkpoints/ettin_gatekeeper \
  --epochs 3 \
  --batch_size 64 \
  --lr 3e-5
```

---

## 4. Evaluation & Testing (`test_gatekeeper.py`)

### Run Automated Duplex Benchmarks:
Tests hesitation pauses, clean turn completions, and barge-in cutoffs:
```bash
uv run python test_gatekeeper.py \
  --benchmark \
  --checkpoint checkpoints/modernbert_gatekeeper/final_best
```

### Run Live Interactive CLI:
Type 1–3 words representing what the mic heard in the last 600 ms (press Enter for `<silence>`):
```bash
uv run python test_gatekeeper.py \
  --checkpoint checkpoints/modernbert_gatekeeper/final_best
```

---

## 5. File Structure Overview

```text
VoiceFullDuplex/
├── pyproject.toml                 # uv package and dependency configuration
├── prepare_gatekeeper_dataset.py  # Converts raw microturns to ModernBERT/Ettin schema
├── download_hf_datasets.py        # Fetches HuggingFace turn-taking datasets
├── train_gatekeeper.py            # Production weighted training script (BF16, metrics)
├── test_gatekeeper.py             # Interactive CLI & benchmark scenario tester
├── FINAL_DUPLEX_MASTER_PLAN.md    # System master execution architecture plan
├── data/
│   ├── gatekeeper_train.jsonl     # Transformed training slices
│   ├── gatekeeper_eval.jsonl      # Transformed validation slices
│   └── hf_downloads/              # Downloaded Hugging Face datasets
└── models/
    └── microturn-lora/            # Base GGUF adapter & raw simulated call datasets
```

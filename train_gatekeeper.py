"""
train_gatekeeper.py

Production training script for the Tier 1 Gatekeeper (ModernBERT-base / Ettin-150m)
with Class-Weighted Cross-Entropy Loss to handle class imbalance (94% LISTEN vs rare INTERRUPTs).

Designed for high-compute machines (e.g., NVIDIA DGX Spark / A100 / RTX 4090 / 3090).
Runnable with uv:
    uv run python train_gatekeeper.py --model_name answerdotai/ModernBERT-base
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

LABEL_NAMES = ["LISTEN", "SPEAK", "FILLER", "INTERRUPT"]


class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies Class-Weighted Cross-Entropy loss.
    This prevents the model from ignoring the rare 'INTERRUPT' (0.8%) and 'SPEAK' (3%) classes.
    """
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Ensure class_weights is on the same device as logits
        weights = self.class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    
    metrics = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    
    # Calculate per-class metrics
    for idx, name in enumerate(LABEL_NAMES):
        class_mask = (labels == idx)
        if np.any(class_mask):
            class_prec = precision_score(labels == idx, preds == idx, zero_division=0)
            class_rec = recall_score(labels == idx, preds == idx, zero_division=0)
            class_f1 = f1_score(labels == idx, preds == idx, zero_division=0)
            metrics[f"f1_{name.lower()}"] = class_f1
            metrics[f"prec_{name.lower()}"] = class_prec
            metrics[f"rec_{name.lower()}"] = class_rec

    return metrics


def calculate_class_weights(dataset_path: str, num_classes: int = 4) -> torch.Tensor:
    """
    Calculates inverse class frequency weights from the training jsonl file.
    """
    counts = np.zeros(num_classes)
    total = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            lbl = item.get("label", 0)
            if 0 <= lbl < num_classes:
                counts[lbl] += 1
                total += 1
                
    print(f"\nTraining Set Class Counts ({total:,} total samples):")
    for i, name in enumerate(LABEL_NAMES):
        pct = (counts[i] / total * 100) if total else 0
        print(f"  Class {i} ({name:<10}): {int(counts[i]):,} ({pct:.2f}%)")
        
    # Inverse class frequency weighting: total / (num_classes * count)
    counts = np.maximum(counts, 1.0)
    weights = total / (num_classes * counts)
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    
    print("\nComputed Class Loss Weights (penalizes mistakes on rare classes):")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  Weight for {name:<10}: {weights[i]:.3f}")
        
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train ModernBERT / Ettin for Duplex Gatekeeper.")
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base",
                        help="HuggingFace model ID (e.g. answerdotai/ModernBERT-base or jhu-clsp/ettin-encoder-150m)")
    parser.add_argument("--train_data", default="data/gatekeeper_train.jsonl")
    parser.add_argument("--eval_data", default="data/gatekeeper_eval.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/gatekeeper_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")

    # 1. Calculate loss weights
    class_weights = calculate_class_weights(args.train_data, num_classes=4)

    # 2. Load Tokenizer & Model
    print(f"\nLoading model & tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=4,
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
        attn_implementation="sdpa"  # Native PyTorch FlashAttention, fully compatible with ARM64 / Grace Hopper
    )

    # 3. Load Datasets
    print(f"Loading datasets from {args.train_data} and {args.eval_data}...")
    raw_datasets = load_dataset("json", data_files={
        "train": args.train_data,
        "eval": args.eval_data
    })

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False  # Collator handles dynamic padding
        )

    print("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"]
    )

    # 4. Training Arguments
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
    use_fp16 = (device == "cuda" and not use_bf16)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=50,
        dataloader_num_workers=2,
        report_to="none"
    )

    # 5. Initialize Trainer (compatible with Transformers 4.x and 5.x)
    import inspect
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["eval"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(
        class_weights=class_weights,
        **trainer_kwargs
    )

    # 6. Train & Evaluate (with Automatic Checkpoint Resumption)
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)

    if last_checkpoint is not None:
        print(f"\n[AUTO-RESUME] Found existing checkpoint: {last_checkpoint}")
        print("Resuming training from this checkpoint (NOT restarting from scratch)...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("\nStarting Training from scratch...")
        trainer.train()

    print("\nEvaluating on Validation Set...")
    eval_results = trainer.evaluate()
    print("\nValidation Metrics:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 7. Final Model Export
    final_save_dir = os.path.join(args.output_dir, "final_best")
    trainer.save_model(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"\nFinal Best Model Saved to: {final_save_dir}")


if __name__ == "__main__":
    main()

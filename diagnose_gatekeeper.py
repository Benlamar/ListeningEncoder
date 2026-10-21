"""
diagnose_gatekeeper.py

Comprehensive evaluation, confusion matrix generator, and threshold calibration tool
for the trained ModernBERT / Ettin Gatekeeper.

Features:
1. Fast batched PyTorch DataLoader inference.
2. Full Confusion Matrix (raw counts and row-normalized percentages).
3. Detailed Classification Report (Precision, Recall, F1 per class).
4. Confidence Threshold Sweep: shows how Precision/Recall trade off when tuning
   P(SPEAK) or P(INTERRUPT) cutoffs.
5. Error Inspector: prints the most confident False Positive mistakes.

Usage:
    uv run python diagnose_gatekeeper.py --checkpoint checkpoints/gatekeeper_model/final_best
    uv run python diagnose_gatekeeper.py --checkpoint checkpoints/gatekeeper_model/final_best --eval-file data/duplexcue_benchmark.jsonl
"""

import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

LABEL_NAMES = ["LISTEN", "SPEAK", "FILLER", "INTERRUPT"]


class JsonlDataset(Dataset):
    def __init__(self, file_path: str, max_samples: int = None):
        self.rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                self.rows.append((item.get("text", ""), int(item.get("label", 0))))
                if max_samples and len(self.rows) >= max_samples:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def run_batched_inference(model, tokenizer, dataset, batch_size=128, device="cuda"):
    def collate_fn(batch):
        texts = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=160, return_tensors="pt")
        return encoded, torch.tensor(labels, dtype=torch.long), texts

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_probs = []
    all_labels = []
    all_texts = []

    print(f"Running inference across {len(dataset):,} samples (batch_size={batch_size})...")
    with torch.no_grad():
        for step, (inputs, labels, texts) in enumerate(loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())
            all_texts.extend(texts)
            if (step + 1) % 20 == 0 or (step + 1) == len(loader):
                print(f"  Processed {min((step + 1) * batch_size, len(dataset)):,} / {len(dataset):,}...")

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    return all_probs, all_labels, all_texts


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (Rows: Ground Truth | Columns: Model Prediction)")
    print("=" * 70)
    header = f"{'True Class':<14} | " + " | ".join([f"{name:>10}" for name in LABEL_NAMES]) + " | Total"
    print(header)
    print("-" * len(header))

    for i, name in enumerate(LABEL_NAMES):
        row_str = " | ".join([f"{cm[i, j]:>6} ({cm_norm[i, j]:4.1f}%)" for j in range(4)])
        print(f"{name:<14} | {row_str} | {cm[i].sum():,}")


def run_threshold_sweep(probs, y_true):
    print("\n" + "=" * 70)
    print("CONFIDENCE THRESHOLD SWEEPS (Fixing Precision without Retraining)")
    print("=" * 70)

    # Sweep for SPEAK (Class 1)
    print("\n>>> Tuning SPEAK Decision Threshold (When AI is IDLE)")
    print(f"{'Threshold':<10} | {'Predicted':<10} | {'True Pos':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 68)

    actual_speak = (y_true == 1)
    total_true_speak = actual_speak.sum()

    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
        pred_speak = (probs[:, 1] >= thresh)
        num_pred = pred_speak.sum()
        tp = (pred_speak & actual_speak).sum()
        prec = (tp / num_pred * 100) if num_pred > 0 else 0.0
        rec = (tp / total_true_speak * 100) if total_true_speak > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        print(f"{thresh:<10.2f} | {num_pred:<10,} | {tp:<10,} | {prec:9.2f}% | {rec:9.2f}% | {f1:9.2f}%")

    # Sweep for INTERRUPT (Class 3)
    print("\n>>> Tuning INTERRUPT Decision Threshold (When AI is SPEAKING)")
    print(f"{'Threshold':<10} | {'Predicted':<10} | {'True Pos':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 68)

    actual_intr = (y_true == 3)
    total_true_intr = actual_intr.sum()

    for thresh in [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        pred_intr = (probs[:, 3] >= thresh)
        num_pred = pred_intr.sum()
        tp = (pred_intr & actual_intr).sum()
        prec = (tp / num_pred * 100) if num_pred > 0 else 0.0
        rec = (tp / total_true_intr * 100) if total_true_intr > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        print(f"{thresh:<10.2f} | {num_pred:<10,} | {tp:<10,} | {prec:9.2f}% | {rec:9.2f}% | {f1:9.2f}%")


def inspect_false_positives(probs, y_true, texts, target_class=1, top_k=10):
    class_name = LABEL_NAMES[target_class]
    print("\n" + "=" * 70)
    print(f"TOP {top_k} MOST CONFIDENT FALSE POSITIVES FOR: {class_name}")
    print("=" * 70)

    # Find cases where model was very confident in target_class, but true label was NOT target_class
    fp_mask = (y_true != target_class)
    confidences = probs[:, target_class]
    fp_indices = np.where(fp_mask)[0]

    # Sort descending by target_class probability
    sorted_fps = fp_indices[np.argsort(-confidences[fp_indices])]

    for rank, idx in enumerate(sorted_fps[:top_k], 1):
        true_lbl = LABEL_NAMES[y_true[idx]]
        pred_prob = confidences[idx]
        all_prob_str = " | ".join([f"{LABEL_NAMES[c]}: {probs[idx, c]:.2f}" for c in range(4)])
        print(f"\n[Error #{rank}] True: {true_lbl} | Predicted {class_name} with {pred_prob*100:.1f}% confidence")
        print(f"  Input: {texts[idx]}")
        print(f"  All Probs: [{all_prob_str}]")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic evaluation for Gatekeeper.")
    parser.add_argument("--checkpoint", default="checkpoints/gatekeeper_model/final_best",
                        help="Path to trained model checkpoint")
    parser.add_argument("--eval-file", default="data/gatekeeper_eval.jsonl",
                        help="Path to validation JSONL file")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    print(f"Loading checkpoint '{args.checkpoint}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=4,
        attn_implementation="sdpa"
    ).to(device)
    model.eval()

    print(f"Loading evaluation dataset from: {args.eval_file}")
    dataset = JsonlDataset(args.eval_file, max_samples=args.max_samples)
    print(f"Loaded {len(dataset):,} evaluation samples.")

    # 1. Batched inference
    probs, y_true, texts = run_batched_inference(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device
    )

    # Standard argmax predictions
    y_pred = np.argmax(probs, axis=1)

    # 2. Confusion Matrix
    print_confusion_matrix(y_true, y_pred)

    # 3. Standard Classification Report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Default Argmax)")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4, zero_division=0))

    # 4. Threshold Sweep
    run_threshold_sweep(probs, y_true)

    # 5. False Positive Inspections
    inspect_false_positives(probs, y_true, texts, target_class=1, top_k=8)  # SPEAK
    inspect_false_positives(probs, y_true, texts, target_class=3, top_k=8)  # INTERRUPT


if __name__ == "__main__":
    main()


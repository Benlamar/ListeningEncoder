"""
unify_gatekeeper_dataset.py

Merges and balances all training data streams into a single unified training dataset:
1. Original Microturn data (data/gatekeeper_train.jsonl)
2. Real-world human OpenYap data (data/openyap_augmented.jsonl)
3. DeepSeek LLM augmented data (data/gatekeeper_llm_augmented.jsonl)

Balancing Strategy:
- Preserves 100% of all rare events: INTERRUPT (Class 3), SPEAK (Class 1), FILLER (Class 2).
- Preserves 100% of LLM thinking-pause hard negatives (Class 0).
- Subsamples repetitive LISTEN silences so the final distribution is ~50% LISTEN, ~20% SPEAK, ~15% FILLER, ~15% INTERRUPT.
- Shuffles everything into a single ready-to-train file.

Usage:
    uv run python unify_gatekeeper_dataset.py
    uv run python unify_gatekeeper_dataset.py --output-file data/gatekeeper_train_unified.jsonl
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any

LABEL_NAMES = {0: "LISTEN", 1: "SPEAK", 2: "FILLER", 3: "INTERRUPT"}


def load_dataset_file(file_path: str, source_name: str) -> Dict[int, List[Dict[str, Any]]]:
    """Loads a JSONL file and groups valid entries by class label."""
    by_class = {0: [], 1: [], 2: [], 3: []}
    if not os.path.exists(file_path):
        print(f"[-] Notice: {source_name} file not found at: {file_path} (Skipping)")
        return by_class

    total = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                lbl = int(item.get("label", -1))
                text = item.get("text", "").strip()
                if 0 <= lbl <= 3 and text:
                    by_class[lbl].append({"text": text, "label": lbl, "source": source_name})
                    total += 1
            except Exception:
                pass

    print(f"[+] Loaded {source_name:<16} ({total:>7,} samples):")
    for lbl in range(4):
        cnt = len(by_class[lbl])
        pct = (cnt / total * 100) if total else 0
        print(f"      Class {lbl} ({LABEL_NAMES[lbl]:<10}): {cnt:>6,} ({pct:5.1f}%)")
    return by_class


def unify_datasets(microturn_path: str,
                   openyap_path: str,
                   llm_path: str,
                   output_path: str,
                   target_listen_ratio: float = 0.52,
                   seed: int = 42):
    random.seed(seed)
    print("=" * 65)
    print("UNIFYING GATEKEEPER TRAINING DATASETS")
    print("=" * 65)

    # 1. Load each data source
    microturn_data = load_dataset_file(microturn_path, "Microturn Base")
    openyap_data = load_dataset_file(openyap_path, "OpenYap Human")
    llm_data = load_dataset_file(llm_path, "DeepSeek LLM")

    # 2. Pool all rare events (Keep 100% of SPEAK, FILLER, INTERRUPT)
    pooled_speak = microturn_data[1] + openyap_data[1] + llm_data[1]
    pooled_filler = microturn_data[2] + openyap_data[2] + llm_data[2]
    pooled_interrupt = microturn_data[3] + openyap_data[3] + llm_data[3]

    num_rare = len(pooled_speak) + len(pooled_filler) + len(pooled_interrupt)
    print("\n" + "-" * 65)
    print(f"Pooled High-Value Minority Events: {num_rare:,} total")
    print(f"  • SPEAK     (Class 1): {len(pooled_speak):,} (100% preserved)")
    print(f"  • FILLER    (Class 2): {len(pooled_filler):,} (100% preserved)")
    print(f"  • INTERRUPT (Class 3): {len(pooled_interrupt):,} (100% preserved)")
    print("-" * 65)

    if num_rare == 0:
        print("ERROR: No minority samples found across sources. Check input paths!")
        return

    # 3. Balance LISTEN (Class 0):
    # To achieve ~target_listen_ratio (e.g. 52% LISTEN, 48% minority classes):
    # target_listen_count = num_rare * (target_listen_ratio / (1 - target_listen_ratio))
    target_listen_count = int(num_rare * (target_listen_ratio / (1.0 - target_listen_ratio)))

    # Priority 1: Keep ALL LLM thinking-pause hard negatives (they cure false SPEAK triggers!)
    kept_listen = list(llm_data[0])

    # Priority 2: Sample remaining LISTEN ticks from Microturn and OpenYap
    remaining_listen_pool = microturn_data[0] + openyap_data[0]
    random.shuffle(remaining_listen_pool)

    remaining_needed = max(0, target_listen_count - len(kept_listen))
    sampled_listen = remaining_listen_pool[:remaining_needed]
    kept_listen.extend(sampled_listen)

    print(f"\nBalanced LISTEN Class:")
    print(f"  • LLM Thinking Negatives: {len(llm_data[0]):,} (100% preserved)")
    print(f"  • Sampled Speech/Silences:{len(sampled_listen):,} (from {len(remaining_listen_pool):,} pool)")
    print(f"  • Total LISTEN Kept:      {len(kept_listen):,}")

    # 4. Combine and Shuffle
    final_dataset = []
    final_dataset.extend(kept_listen)
    final_dataset.extend(pooled_speak)
    final_dataset.extend(pooled_filler)
    final_dataset.extend(pooled_interrupt)

    random.shuffle(final_dataset)

    # 5. Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in final_dataset:
            counts[item["label"]] += 1
            fout.write(json.dumps({"text": item["text"], "label": item["label"]}) + "\n")

    total_final = len(final_dataset)
    print("\n" + "=" * 65)
    print(f"UNIFIED DATASET SUCCESSFULLY SAVED TO: {output_path}")
    print(f"Total Samples: {total_final:,}")
    print("=" * 65)
    for lbl in range(4):
        cnt = counts[lbl]
        pct = (cnt / total_final * 100) if total_final else 0
        print(f"  Class {lbl} ({LABEL_NAMES[lbl]:<10}): {cnt:>6,} ({pct:5.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Unify and balance Gatekeeper training datasets.")
    parser.add_argument("--microturn-data", default="data/gatekeeper_train.jsonl",
                        help="Path to original microturn training data")
    parser.add_argument("--openyap-data", default="data/openyap_augmented.jsonl",
                        help="Path to real human OpenYap data")
    parser.add_argument("--llm-data", default="data/gatekeeper_llm_augmented.jsonl",
                        help="Path to LLM augmented data")
    parser.add_argument("--output-file", default="data/gatekeeper_train_unified.jsonl",
                        help="Output path for unified dataset")
    parser.add_argument("--listen-ratio", type=float, default=0.52,
                        help="Target ratio of LISTEN class in final dataset (default: 0.52)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    unify_datasets(
        microturn_path=args.microturn_data,
        openyap_path=args.openyap_data,
        llm_path=args.llm_data,
        output_path=args.output_file,
        target_listen_ratio=args.listen_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

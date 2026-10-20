"""
prepare_gatekeeper_dataset.py

Converts the 404,000 micro-turn ticks from models/microturn-lora/dataset/
into ModernBERT / Ettin sequence classification format:
  text:  "AI_State: <IDLE|SPEAKING> | History: <recent context> | Slice: <new words>"
  label: 0 (LISTEN), 1 (SPEAK), 2 (FILLER), 3 (INTERRUPT)
"""

import json
import os
import argparse
from typing import Dict, List, Tuple

# Mapping of raw microturn tags to ModernBERT's 4 classes
TAG_MAP = {
    # 0: LISTEN (User talking, thinking pause, idle silence, or assistant continuing speech)
    "listen_speech": 0,
    "listen_pause": 0,
    "listen_idle": 0,
    "listen_last": 0,
    "listen_done": 0,
    "barge_listen": 0,
    "speak": 0,          # Assistant is already speaking; stay in listen/stream mode
    "hazard": 0,         # Assistant is continuing a warning

    # 1: SPEAK (Turn complete, trigger LLM generation)
    "start": 1,

    # 2: FILLER (Backchannel nod, e.g. "Mm-hm", "Right")
    "continuer": 2,

    # 3: INTERRUPT (Barge-in while speaking or immediate safety hazard)
    "barge_stop": 3,
    "hazard_start": 3,
}

LABEL_NAMES = {0: "LISTEN", 1: "SPEAK", 2: "FILLER", 3: "INTERRUPT"}


def process_call(call_obj: dict, max_history_words: int = 40) -> List[dict]:
    """
    Simulates a call tick by tick, building a sliding window history
    and producing labeled rows for ModernBERT.
    """
    rows = []
    ticks = call_obj.get("ticks", [])
    
    # State tracking
    user_buffer = []      # recent user words
    assistant_buffer = [] # recent assistant spoken words
    is_ai_speaking = False
    
    for t in ticks:
        heard = t.get("heard", "").strip()
        said = t.get("said", "").strip()
        tag = t.get("tag", "")
        
        # Determine class label
        label = TAG_MAP.get(tag, 0)
        
        # Convert audio notation: <no voice> -> <silence>
        slice_text = "<silence>" if (heard == "<no voice>" or not heard) else heard

        # Build AI_State string
        if is_ai_speaking:
            recent_ai = " ".join(assistant_buffer[-10:])
            ai_state = f"SPEAKING (AI: {recent_ai})" if recent_ai else "SPEAKING"
        else:
            ai_state = "IDLE"

        # Build History string (bounded to recent words to keep inference fast)
        recent_user = " ".join(user_buffer[-max_history_words:])
        history_str = f"User: {recent_user}" if recent_user else ""

        # ModernBERT prompt input
        prompt_text = f"AI_State: {ai_state} | History: {history_str} | Slice: {slice_text}"
        
        rows.append({
            "text": prompt_text,
            "label": label,
            "raw_tag": tag
        })

        # Update state for next tick
        if heard and heard != "<no voice>":
            user_buffer.append(heard)
            
        if said and said != "<listen>":
            is_ai_speaking = True
            assistant_buffer.append(said)
        else:
            if tag in ["listen_done", "listen_idle", "barge_stop"]:
                is_ai_speaking = False
                assistant_buffer = []

    return rows


def convert_dataset(input_file: str, output_file: str, max_calls: int = None, max_history: int = 40):
    print(f"Reading from {input_file}...")
    total_calls = 0
    total_rows = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            call_obj = json.loads(line)
            rows = process_call(call_obj, max_history_words=max_history)
            
            for r in rows:
                class_counts[r["label"]] += 1
                fout.write(json.dumps({"text": r["text"], "label": r["label"]}) + "\n")
                total_rows += 1

            total_calls += 1
            if max_calls and total_calls >= max_calls:
                break

    print(f"Done! Processed {total_calls} calls into {total_rows} rows -> {output_file}")
    print("Class Distribution:")
    for lbl, cnt in class_counts.items():
        pct = (cnt / total_rows * 100) if total_rows else 0
        print(f"  Class {lbl} ({LABEL_NAMES[lbl]}): {cnt:,} ({pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Convert microturn calls to ModernBERT dataset format.")
    parser.add_argument("--train-in", default="models/microturn-lora/dataset/calls_train.jsonl")
    parser.add_argument("--eval-in", default="models/microturn-lora/dataset/calls_heldout.jsonl")
    parser.add_argument("--train-out", default="data/gatekeeper_train.jsonl")
    parser.add_argument("--eval-out", default="data/gatekeeper_eval.jsonl")
    parser.add_argument("--smoke-test", action="store_true", help="Generate a small ~5,000-row dataset for quick smoke test")
    args = parser.parse_args()

    max_train_calls = 20 if args.smoke_test else None
    max_eval_calls = 5 if args.smoke_test else None

    print(f"--- Converting Training Data (smoke_test={args.smoke_test}) ---")
    convert_dataset(args.train_in, args.train_out, max_calls=max_train_calls)

    print(f"\n--- Converting Evaluation Data (smoke_test={args.smoke_test}) ---")
    convert_dataset(args.eval_in, args.eval_out, max_calls=max_eval_calls)


if __name__ == "__main__":
    main()

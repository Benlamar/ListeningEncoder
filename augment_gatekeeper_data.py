"""
augment_gatekeeper_data.py

Balances the Gatekeeper dataset by:
1. Subsampling redundant LISTEN ticks (e.g. repetitive idle/assistant-speaking silences).
2. Generating diverse, high-leverage barge-in triggers (INTERRUPT - Class 3) over active AI speech.
3. Generating hard-negative thinking pauses ending with <silence> (LISTEN - Class 0) to fix false SPEAK triggers.
4. Generating clean turn-completion questions & statements ending with <silence> (SPEAK - Class 1).
5. Generating conversational nods & backchannels (FILLER - Class 2).

Usage:
    uv run python augment_gatekeeper_data.py
    uv run python augment_gatekeeper_data.py --train-in data/gatekeeper_train.jsonl --train-out data/gatekeeper_train_augmented.jsonl
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any

LABEL_NAMES = {0: "LISTEN", 1: "SPEAK", 2: "FILLER", 3: "INTERRUPT"}

# Diverse barge-in phrases representing real human cut-ins across 1-3 word streaming STT slices
BARGE_IN_PHRASES = [
    # Direct stops
    "Wait", "Stop", "Hold on", "Hang on", "Hold up", "Wait wait", "Stop stop",
    "Wait a second", "Hold on a second", "Wait a minute", "Wait, hold on",
    "Wait, stop", "Just a second", "One second", "Just a moment", "Hold on please",
    # Clarifications & questions
    "Wait what?", "Wait, what?", "Excuse me", "Sorry to interrupt", "Can I ask something?",
    "Quick question", "Can I stop you?", "Wait, which one?", "Wait, does it",
    "Wait, are you sure?", "Wait, what do you mean?", "Wait, really?", "Wait, why?",
    # Corrections & objections
    "No wait", "No actually", "No, actually", "No, that's not right", "No no no",
    "Actually, before you continue", "Wait, that's wrong", "Sorry, but", "Not really",
    "Wait, I meant", "Actually wait", "Wait, don't do that", "Wait, listen"
]

# Hard negative thinking pauses: user pauses mid-thought, ending with <silence>
# True label MUST be LISTEN (Class 0), NOT SPEAK!
THINKING_INCOMPLETE_ENDINGS = [
    "I was thinking about",
    "because I don't really",
    "and then we should probably",
    "or maybe if you could",
    "so if that's possible then",
    "but actually what happened was",
    "um, let me see if I can",
    "I wanted to ask whether or not",
    "at the same time as the",
    "well I'm not really sure if",
    "so like, for example,",
    "and I was looking at the",
    "even though it was kind of",
    "since we already know that",
    "although I don't think",
    "which means that probably",
    "after I get back from the",
    "before we actually decide to",
    "just in case you happen to",
    "especially when it comes to",
    "unless there's another way to",
    "as long as it doesn't take too",
    "you know, kind of like",
    "and uh, what was I going to say,",
    "wait let me check my",
    "hang on a second, looking at",
    "I guess it depends on whether",
    "to be completely honest with you,"
]

# Clean completed questions and statements followed by <silence> (SPEAK - Class 1)
TURN_COMPLETION_PATTERNS = [
    "Can you tell me what time the store opens?",
    "What is the best way to handle this?",
    "How much does that usually cost?",
    "Could you help me reset my password?",
    "Where should I send the confirmation email?",
    "Is there anything else I need to bring with me?",
    "When will the package arrive?",
    "Can you explain that one more time?",
    "Does this cover water damage as well?",
    "What should I do if that doesn't work?",
    "That makes total sense, thank you.",
    "Yes, please go ahead and process that.",
    "No, that will be all for today.",
    "Sounds great, let's do that.",
    "I'm ready for the next step now.",
    "Okay, I've got that written down.",
    "Got it, thanks for explaining."
]

# Backchannel phrases (FILLER - Class 2)
BACKCHANNEL_WORDS = [
    "Mm-hm.", "Yeah.", "Right.", "Okay.", "Got it.", "Uh-huh.", "Sure.",
    "I see.", "Makes sense.", "Exactly.", "Understood."
]


def load_dialogue_texts(dialogues_path: str) -> List[Dict[str, Any]]:
    """Loads dialogue turns to sample realistic AI states and User histories."""
    dialogues = []
    if not os.path.exists(dialogues_path):
        return dialogues
    with open(dialogues_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    dialogues.append(json.loads(line))
                except Exception:
                    pass
    return dialogues


def generate_barge_in_samples(dialogues: List[Dict[str, Any]], count: int = 6000) -> List[Dict[str, Any]]:
    """
    Generates synthetic barge-in ticks (INTERRUPT - Class 3) over active AI speech.
    Crucial for fixing the 83:1 anti-interrupt bias!
    """
    samples = []
    ai_phrases = []

    # Harvest real assistant utterances from written dialogues
    for d in dialogues:
        for turn in d.get("turns", []):
            asst = turn.get("assistant", "").strip()
            if asst and len(asst.split()) >= 4:
                ai_phrases.append(asst)

    # Fallback assistant utterances if dialogues aren't loaded
    if not ai_phrases:
        ai_phrases = [
            "Modern chargers usually stop automatically when the battery is full.",
            "You can find the settings menu right under account preferences.",
            "I would recommend checking the tire pressure before long trips.",
            "To qualify for the discount, you need to sign up for annual billing.",
            "The technician will be arriving between nine and eleven tomorrow morning.",
            "Make sure to keep the receipt in case you need to make an exchange.",
            "Our servers will undergo scheduled maintenance tonight at midnight.",
            "You can easily transfer your data using the backup utility."
        ]

    for _ in range(count):
        full_ai = random.choice(ai_phrases)
        ai_words = full_ai.split()
        
        # Simulate AI in the middle of speaking (between 2 and 12 words in)
        cut_len = random.randint(2, min(12, len(ai_words)))
        spoken_so_far = " ".join(ai_words[:cut_len])
        ai_state = f"SPEAKING (AI: {spoken_so_far})"

        # Slice is a natural human cut-in phrase
        cut_in_phrase = random.choice(BARGE_IN_PHRASES)

        # Recent history before AI started speaking
        hist_options = [
            "User: Can you explain how this works?",
            "User: How do I fix the error code?",
            "User: Where can I download the update?",
            "User: Tell me about the warranty policy.",
            ""
        ]
        history = random.choice(hist_options)

        prompt = f"AI_State: {ai_state} | History: {history} | Slice: {cut_in_phrase}"
        samples.append({
            "text": prompt,
            "label": 3,  # INTERRUPT
            "augmented_type": "barge_in"
        })

    return samples


def generate_thinking_pause_samples(count: int = 4000) -> List[Dict[str, Any]]:
    """
    Generates hard-negative thinking pauses ending in <silence> (LISTEN - Class 0).
    Teaches the model NOT to blindly predict SPEAK on <silence> when sentence is incomplete.
    """
    samples = []
    for _ in range(count):
        ending = random.choice(THINKING_INCOMPLETE_ENDINGS)
        # Random prefix to vary sentence structure
        prefixes = [
            "Well,", "Honestly,", "Yesterday,", "So basically,", "At first,",
            "I'm trying to figure out,", "Regarding that,", ""
        ]
        prefix = random.choice(prefixes)
        clause = f"{prefix} {ending}".strip()

        prompt = f"AI_State: IDLE | History: User: {clause} | Slice: <silence>"
        samples.append({
            "text": prompt,
            "label": 0,  # LISTEN (Do not jump in!)
            "augmented_type": "thinking_pause_hard_negative"
        })
    return samples


def generate_turn_completion_samples(count: int = 4000) -> List[Dict[str, Any]]:
    """Generates clean turn completions followed by <silence> (SPEAK - Class 1)."""
    samples = []
    for _ in range(count):
        sentence = random.choice(TURN_COMPLETION_PATTERNS)
        prompt = f"AI_State: IDLE | History: User: {sentence} | Slice: <silence>"
        samples.append({
            "text": prompt,
            "label": 1,  # SPEAK
            "augmented_type": "turn_completion"
        })
    return samples


def generate_backchannel_samples(count: int = 3000) -> List[Dict[str, Any]]:
    """Generates conversational backchannels (FILLER - Class 2)."""
    samples = []
    histories = [
        "User: So I went down to the store and they told me it was completely out of stock and then",
        "User: We were talking about the project deadlines and how we need to allocate more resources to",
        "User: First you open the terminal and then you run the package manager update script before you",
        "User: I've been practicing every day for twenty minutes and my teacher said that next month"
    ]
    for _ in range(count):
        hist = random.choice(histories)
        word = random.choice(BACKCHANNEL_WORDS)
        prompt = f"AI_State: IDLE | History: {hist} | Slice: {word}"
        samples.append({
            "text": prompt,
            "label": 2,  # FILLER
            "augmented_type": "backchannel"
        })
    return samples


def augment_and_balance_dataset(train_in: str,
                                train_out: str,
                                dialogues_paths: List[str],
                                listen_subsample_ratio: float = 0.25,
                                num_barge_ins: int = 6000,
                                num_thinking_negatives: int = 4000,
                                num_turn_completions: int = 4000,
                                num_backchannels: int = 3000,
                                seed: int = 42):
    random.seed(seed)

    print(f"Loading dialogues from: {dialogues_paths}...")
    dialogues = []
    for p in dialogues_paths:
        dialogues.extend(load_dialogue_texts(p))
    print(f"Loaded {len(dialogues)} dialogues for context generation.")

    # 1. Read existing training data and filter/subsample
    print(f"\nReading raw training data from {train_in}...")
    existing_by_class = {0: [], 1: [], 2: [], 3: []}
    total_raw = 0

    with open(train_in, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            lbl = item.get("label", 0)
            existing_by_class[lbl].append(item)
            total_raw += 1

    print(f"Original Training Distribution ({total_raw:,} total):")
    for lbl in range(4):
        cnt = len(existing_by_class[lbl])
        pct = (cnt / total_raw * 100) if total_raw else 0
        print(f"  Class {lbl} ({LABEL_NAMES[lbl]:<10}): {cnt:,} ({pct:.2f}%)")

    # Subsample LISTEN (Class 0):
    # Keep ALL speech/pause ticks, but subsample repetitive idle/speaking silences
    raw_listen = existing_by_class[0]
    kept_listen = []
    for item in raw_listen:
        text = item.get("text", "")
        # If it's a repetitive pure silence tick while AI speaks or is idle, subsample
        if "Slice: <silence>" in text or "Slice: <no voice>" in text:
            if random.random() < listen_subsample_ratio:
                kept_listen.append(item)
        else:
            # Active user speech ticks are valuable, keep at higher rate
            if random.random() < (listen_subsample_ratio * 1.5):
                kept_listen.append(item)

    print(f"\nSubsampled LISTEN from {len(raw_listen):,} down to {len(kept_listen):,}")

    # 2. Generate Augmentations
    print(f"\nSynthesizing high-leverage samples:")
    barge_ins = generate_barge_in_samples(dialogues, count=num_barge_ins)
    print(f"  + Generated {len(barge_ins):,} Barge-In triggers (INTERRUPT - Class 3)")

    thinking_pauses = generate_thinking_pause_samples(count=num_thinking_negatives)
    print(f"  + Generated {len(thinking_pauses):,} Thinking Pause hard negatives (LISTEN - Class 0)")

    completions = generate_turn_completion_samples(count=num_turn_completions)
    print(f"  + Generated {len(completions):,} Turn Completions (SPEAK - Class 1)")

    backchannels = generate_backchannel_samples(count=num_backchannels)
    print(f"  + Generated {len(backchannels):,} Backchannels (FILLER - Class 2)")

    # 3. Combine into final dataset
    final_dataset = []
    # Kept existing samples
    final_dataset.extend(kept_listen)
    final_dataset.extend(existing_by_class[1])  # Keep ALL existing SPEAK
    final_dataset.extend(existing_by_class[2])  # Keep ALL existing FILLER
    final_dataset.extend(existing_by_class[3])  # Keep ALL existing INTERRUPT
    # Augmented samples
    final_dataset.extend(barge_ins)
    final_dataset.extend(thinking_pauses)
    final_dataset.extend(completions)
    final_dataset.extend(backchannels)

    # Shuffle
    random.shuffle(final_dataset)

    # 4. Write output
    os.makedirs(os.path.dirname(train_out) or ".", exist_ok=True)
    final_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    with open(train_out, "w", encoding="utf-8") as fout:
        for item in final_dataset:
            final_counts[item["label"]] += 1
            fout.write(json.dumps({"text": item["text"], "label": item["label"]}) + "\n")

    total_final = len(final_dataset)
    print(f"\n{'='*60}")
    print(f"BALANCED AUGMENTED DATASET SAVED TO: {train_out}")
    print(f"Total samples: {total_final:,} (was {total_raw:,})")
    print(f"{'='*60}")
    for lbl in range(4):
        cnt = final_counts[lbl]
        pct = (cnt / total_final * 100) if total_final else 0
        print(f"  Class {lbl} ({LABEL_NAMES[lbl]:<10}): {cnt:,} ({pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Balance and augment Gatekeeper training data.")
    parser.add_argument("--train-in", default="data/gatekeeper_train.jsonl")
    parser.add_argument("--train-out", default="data/gatekeeper_train_augmented.jsonl")
    parser.add_argument("--dialogues", nargs="+", default=[
        "models/microturn-lora/dataset/dialogues_short.jsonl",
        "models/microturn-lora/dataset/dialogues_long.jsonl"
    ])
    parser.add_argument("--subsample-listen", type=float, default=0.25,
                        help="Subsampling ratio for repetitive LISTEN ticks (default: 0.25)")
    parser.add_argument("--barge-ins", type=int, default=6000, help="Number of synthetic barge-ins to generate")
    parser.add_argument("--thinking-negatives", type=int, default=4000, help="Number of thinking pause hard negatives")
    parser.add_argument("--turn-completions", type=int, default=4000, help="Number of turn completions to generate")
    parser.add_argument("--backchannels", type=int, default=3000, help="Number of backchannels to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    augment_and_balance_dataset(
        train_in=args.train_in,
        train_out=args.train_out,
        dialogues_paths=args.dialogues,
        listen_subsample_ratio=args.subsample_listen,
        num_barge_ins=args.barge_ins,
        num_thinking_negatives=args.thinking_negatives,
        num_turn_completions=args.turn_completions,
        num_backchannels=args.backchannels,
        seed=args.seed
    )


if __name__ == "__main__":
    main()


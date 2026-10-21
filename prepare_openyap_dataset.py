"""
prepare_openyap_dataset.py

Extracts natural human turn-taking, overlaps, and backchannels from
mrfakename/open-yap-1k (in data/hf_downloads/mrfakename_open-yap-1k/)
and converts them into Gatekeeper 600 ms sequence classification ticks:
  text:  "AI_State: <IDLE|SPEAKING> | History: <context> | Slice: <new words>"
  label: 0 (LISTEN), 1 (SPEAK), 2 (FILLER), 3 (INTERRUPT)

Usage:
    uv run python prepare_openyap_dataset.py
    uv run python prepare_openyap_dataset.py --output-file data/openyap_augmented.jsonl
"""

import os
import tarfile
import json
import argparse
from typing import List, Dict, Any

LABEL_NAMES = {0: "LISTEN", 1: "SPEAK", 2: "FILLER", 3: "INTERRUPT"}
BACKCHANNEL_WORDS = {"yeah", "uh-huh", "mm-hm", "right", "okay", "yep", "sure", "mhm", "cool"}


def extract_openyap_tar_if_needed(sample_tar: str, extract_dir: str):
    if not os.path.exists(sample_tar):
        return
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        return
    print(f"Extracting sample conversations from {sample_tar} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(sample_tar, "r") as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete.")


def load_words(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("words", [])
        except Exception:
            return []
    return []


def process_dual_track_conv(user_words: List[Dict[str, Any]],
                            ai_words: List[Dict[str, Any]],
                            tick_duration: float = 0.6,
                            max_history_words: int = 35) -> List[Dict[str, Any]]:
    rows = []
    if not user_words and not ai_words:
        return rows

    # Normalize structure to have start, end, word
    def get_time(item, key, default):
        return float(item.get(key, default))

    user_words.sort(key=lambda x: get_time(x, "start", 0.0))
    ai_words.sort(key=lambda x: get_time(x, "start", 0.0))

    max_time = 0.0
    for w in user_words + ai_words:
        max_time = max(max_time, get_time(w, "end", 0.0))

    user_history = []
    ai_buffer = []

    t = 0.0
    while t < max_time:
        t_next = t + tick_duration

        # Words in this tick
        u_tick = [w.get("word", "").strip() for w in user_words if t <= (get_time(w, "start", 0) + get_time(w, "end", 0)) / 2 < t_next and w.get("word", "").strip()]
        a_tick = [w.get("word", "").strip() for w in ai_words if t <= (get_time(w, "start", 0) + get_time(w, "end", 0)) / 2 < t_next and w.get("word", "").strip()]

        u_active = any(get_time(w, "start", 0) < t_next and get_time(w, "end", 0) > t for w in user_words)
        a_active = any(get_time(w, "start", 0) < t_next and get_time(w, "end", 0) > t for w in ai_words)

        if a_active:
            ai_str = "SPEAKING (AI: " + " ".join(ai_buffer[-10:]) + ")" if ai_buffer else "SPEAKING"
        else:
            ai_str = "IDLE"

        hist_str = "User: " + " ".join(user_history[-max_history_words:]) if user_history else ""
        slice_str = " ".join(u_tick) if u_tick else "<silence>"

        # Determine label
        # Case 1: AI is speaking and User cuts in
        if a_active and u_active and u_tick:
            # Check if it's just a backchannel or actual barge-in
            is_backchannel = all(w.lower().strip(".,?!") in BACKCHANNEL_WORDS for w in u_tick)
            label = 2 if is_backchannel else 3
        # Case 2: AI is idle, user just finished speaking in previous tick and now silence
        elif not a_active and not u_active and user_history:
            # If silence right after speech, potential turn completion (SPEAK)
            label = 1
        elif not a_active and u_active:
            # User speaking
            label = 0
        else:
            label = 0

        prompt = f"AI_State: {ai_str} | History: {hist_str} | Slice: {slice_str}"
        rows.append({"text": prompt, "label": label})

        # Update rolling state
        if u_tick:
            user_history.extend(u_tick)
        if a_tick:
            ai_buffer.extend(a_tick)
        if not a_active:
            ai_buffer = []

        t = t_next

    return rows


def main():
    parser = argparse.ArgumentParser(description="Convert OpenYap 1K conversations to Gatekeeper training format.")
    parser.add_argument("--source-dir", default="data/hf_downloads/mrfakename_open-yap-1k")
    parser.add_argument("--output-file", default="data/openyap_augmented.jsonl")
    parser.add_argument("--max-conversations", type=int, default=16)
    args = parser.parse_args()

    sample_tar = os.path.join(args.source_dir, "open-yap-1k-sample.tar")
    conv_dir = os.path.join(args.source_dir, "extracted_sample")
    extract_openyap_tar_if_needed(sample_tar, conv_dir)

    if not os.path.exists(conv_dir):
        print(f"Notice: Extracted OpenYap directory '{conv_dir}' not found.")
        print(f"Run: uv run python download_hf_datasets.py --repo mrfakename/open-yap-1k")
        return

    print(f"Scanning {conv_dir} for dual-track conversations...")
    all_rows = []
    # Process conversations
    subdirs = [os.path.join(conv_dir, d) for d in os.listdir(conv_dir) if os.path.isdir(os.path.join(conv_dir, d))]
    
    count = 0
    for sd in subdirs[:args.max_conversations]:
        words_a = load_words(os.path.join(sd, "a_words.json")) or load_words(os.path.join(sd, "transcript_a.json"))
        words_b = load_words(os.path.join(sd, "b_words.json")) or load_words(os.path.join(sd, "transcript_b.json"))

        if words_a and words_b:
            # Perspective 1: A is AI, B is User
            r1 = process_dual_track_conv(user_words=words_b, ai_words=words_a)
            # Perspective 2: B is AI, A is User
            r2 = process_dual_track_conv(user_words=words_a, ai_words=words_b)
            all_rows.extend(r1)
            all_rows.extend(r2)
            count += 1

    print(f"Converted {count} dual-perspective conversations into {len(all_rows):,} ticks.")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()


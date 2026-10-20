"""
prepare_duplexcue_dataset.py

Converts the 2-person human conversational audio-aligned transcripts
from data/hf_downloads/besimple-ai_duplex-cue/ into ModernBERT 600 ms tick format.

Extracts real-world human:
  - Backchannels ("mm-hm", "yeah") -> Class 2 (FILLER)
  - Barge-in interruptions while speaker holds floor -> Class 3 (INTERRUPT)
  - Turn handoffs / completions -> Class 1 (SPEAK)
  - Ongoing speech and thinking pauses -> Class 0 (LISTEN)
"""

import os
import json
import argparse
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items


def is_active(items: List[Dict[str, Any]], t_start: float, t_end: float) -> bool:
    """Checks if a speaker spoke during [t_start, t_end)."""
    for item in items:
        start = item.get("start", 0.0)
        end = item.get("end", 0.0)
        if start < t_end and end > t_start:
            return True
        if start >= t_end:
            break
    return False


def get_words_in_window(words: List[Dict[str, Any]], t_start: float, t_end: float) -> List[str]:
    """Returns words spoken within the time window [t_start, t_end)."""
    matched = []
    for w in words:
        start = w.get("start", 0.0)
        end = w.get("end", 0.0)
        word_text = w.get("word", "").strip()
        # Word midpoint falls into the tick
        mid = (start + end) / 2.0
        if t_start <= mid < t_end:
            if word_text:
                matched.append(word_text)
        elif start >= t_end:
            break
    return matched


def process_conversation_perspective(conv_id: str,
                                     user_track: int,
                                     ai_track: int,
                                     conv_dir: str,
                                     tick_duration: float = 0.6,
                                     max_history_words: int = 40) -> List[Dict[str, Any]]:
    tracks_dir = os.path.join(conv_dir, "transcription", "tracks")
    user_words = load_jsonl(os.path.join(tracks_dir, f"words.{user_track}.jsonl"))
    ai_words = load_jsonl(os.path.join(tracks_dir, f"words.{ai_track}.jsonl"))
    user_segs = load_jsonl(os.path.join(tracks_dir, f"segments.{user_track}.jsonl"))
    ai_segs = load_jsonl(os.path.join(tracks_dir, f"segments.{ai_track}.jsonl"))
    candidate_trials = load_jsonl(os.path.join(conv_dir, "candidate_trials.jsonl"))

    if not user_words or not ai_words:
        return []

    # Sort items by start time
    user_words.sort(key=lambda x: x.get("start", 0.0))
    ai_words.sort(key=lambda x: x.get("start", 0.0))
    user_segs.sort(key=lambda x: x.get("start", 0.0))
    ai_segs.sort(key=lambda x: x.get("start", 0.0))

    # Pre-index cue events (interruptions, backchannels) for this perspective
    # Note: in duplex-cue, output_track is the ongoing speaker, input_track is the person cutting in
    cues_by_time = []
    for ct in candidate_trials:
        cue_type = ct.get("cue_type", "")
        cue_time = ct.get("cue_start_time", 0.0)
        input_track_name = ct.get("input_track", "")
        output_track_name = ct.get("output_track", "")
        
        # Check if the user is the one delivering the cue into the AI's turn
        if f"track_{user_track}" in input_track_name and f"track_{ai_track}" in output_track_name:
            cues_by_time.append((cue_time, cue_type))

    # Calculate conversation end time
    max_time = max(
        user_words[-1].get("end", 0.0) if user_words else 0.0,
        ai_words[-1].get("end", 0.0) if ai_words else 0.0
    )

    rows = []
    user_history = []
    ai_history = []
    t = 0.0

    while t < max_time:
        t_next = t + tick_duration

        # 1. User words in this tick
        slice_words = get_words_in_window(user_words, t, t_next)
        slice_text = " ".join(slice_words) if slice_words else "<silence>"

        # 2. Is AI currently speaking?
        ai_speaking = is_active(ai_segs, t, t_next)
        if ai_speaking:
            recent_ai = " ".join(ai_history[-10:])
            ai_state = f"SPEAKING (AI: {recent_ai})" if recent_ai else "SPEAKING"
        else:
            ai_state = "IDLE"

        # 3. Class decision logic
        # Default to LISTEN (0)
        label = 0

        # Check for annotated cues (interruption or backchannel)
        for cue_t, cue_type in cues_by_time:
            if t <= cue_t < t_next:
                if cue_type == "interruption" and ai_speaking:
                    label = 3  # INTERRUPT
                elif cue_type in ["backchannel", "collaboration"]:
                    label = 2  # FILLER

        # Check for clean turn transition (User finishes and AI starts shortly after)
        if label == 0 and not ai_speaking and slice_text == "<silence>":
            # Was user active in the last 1.2 seconds, and does AI start speaking in the next 1.2 seconds?
            user_just_finished = is_active(user_segs, max(0.0, t - 1.2), t)
            ai_about_to_speak = is_active(ai_segs, t_next, t_next + 1.2)
            if user_just_finished and ai_about_to_speak:
                label = 1  # SPEAK (Turn Complete)

        # 4. Build prompt string
        history_str = f"User: {' '.join(user_history[-max_history_words:])}" if user_history else ""
        prompt_text = f"AI_State: {ai_state} | History: {history_str} | Slice: {slice_text}"

        rows.append({
            "text": prompt_text,
            "label": label,
            "meta": {"conv": conv_id, "time": round(t, 2), "perspective": f"user_track_{user_track}"}
        })

        # Update histories
        if slice_words:
            user_history.extend(slice_words)
        ai_words_now = get_words_in_window(ai_words, t, t_next)
        if ai_words_now:
            ai_history.extend(ai_words_now)

        t = t_next

    return rows


def process_duplexcue(base_dir: str, output_file: str):
    convs_dir = os.path.join(base_dir, "conversations")
    if not os.path.exists(convs_dir):
        print(f"Error: {convs_dir} not found.")
        return

    conv_names = sorted([d for d in os.listdir(convs_dir) if os.path.isdir(os.path.join(convs_dir, d))])
    print(f"Found {len(conv_names)} conversations in Duplex-Cue: {conv_names}")

    all_rows = []
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    label_names = {0: "LISTEN", 1: "SPEAK", 2: "FILLER", 3: "INTERRUPT"}

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    for conv in conv_names:
        c_path = os.path.join(convs_dir, conv)
        # Process Perspective 1: Track 0 as User, Track 1 as AI
        rows_0 = process_conversation_perspective(conv, user_track=0, ai_track=1, conv_dir=c_path)
        # Process Perspective 2: Track 1 as User, Track 0 as AI
        rows_1 = process_conversation_perspective(conv, user_track=1, ai_track=0, conv_dir=c_path)

        all_rows.extend(rows_0)
        all_rows.extend(rows_1)
        print(f"  Processed {conv}: {len(rows_0) + len(rows_1)} slices")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in all_rows:
            class_counts[r["label"]] += 1
            f.write(json.dumps({"text": r["text"], "label": r["label"]}) + "\n")

    print(f"\nSuccessfully generated {len(all_rows):,} slices -> {output_file}")
    print("Class Distribution:")
    for lbl, cnt in class_counts.items():
        pct = (cnt / len(all_rows) * 100) if all_rows else 0
        print(f"  Class {lbl} ({label_names[lbl]:<10}): {cnt:,} ({pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Convert Duplex-Cue human benchmark to ModernBERT format.")
    parser.add_argument("--input-dir", default="data/hf_downloads/besimple-ai_duplex-cue")
    parser.add_argument("--output-file", default="data/duplexcue_benchmark.jsonl")
    args = parser.parse_args()

    process_duplexcue(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()

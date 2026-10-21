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
from typing import List, Dict, Any, Tuple

LABEL_NAMES = {0: "LISTEN", 1: "SPEAK", 2: "FILLER", 3: "INTERRUPT"}
BACKCHANNEL_WORDS = {"yeah", "uh-huh", "mm-hm", "right", "okay", "yep", "sure", "mhm", "cool", "got it"}


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


def extract_words_from_deepgram_or_raw(data: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracts (words_track_a, words_track_b) from whatever schema OpenYap provides:
    1. Multi-channel deepgram: results.channels[0] & results.channels[1]
    2. Diarized words: words with speaker: 0 and speaker: 1
    3. Flat word lists
    """
    words_a = []
    words_b = []

    if isinstance(data, dict):
        # Format 0: OpenYap 1K official schema: transcripts['a']['words'] & transcripts['b']['words']
        transcripts = data.get("transcripts", {})
        if isinstance(transcripts, dict) and "a" in transcripts and "b" in transcripts:
            words_a = transcripts["a"].get("words", [])
            words_b = transcripts["b"].get("words", [])
            if words_a or words_b:
                return words_a, words_b

        # Format A: Deepgram channels object
        channels = data.get("results", {}).get("channels", [])
        if len(channels) >= 2:
            alt_a = channels[0].get("alternatives", [{}])[0].get("words", [])
            alt_b = channels[1].get("alternatives", [{}])[0].get("words", [])
            return alt_a, alt_b
        elif len(channels) == 1:
            alt = channels[0].get("alternatives", [{}])[0].get("words", [])
            # Check speaker tags
            for w in alt:
                spk = w.get("speaker", 0)
                if spk == 0:
                    words_a.append(w)
                else:
                    words_b.append(w)
            if words_a or words_b:
                return words_a, words_b

        # Format B: Direct words list with speaker field
        words = data.get("words", [])
        if words and isinstance(words, list):
            for w in words:
                spk = w.get("speaker", 0)
                if spk == 0:
                    words_a.append(w)
                else:
                    words_b.append(w)
            return words_a, words_b

        # Format C: Track a / b keys
        if "track_a" in data and "track_b" in data:
            return data["track_a"].get("words", []), data["track_b"].get("words", [])

    return words_a, words_b


def load_single_file_words(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if "words" in data:
                    return data["words"]
                channels = data.get("results", {}).get("channels", [])
                if channels:
                    return channels[0].get("alternatives", [{}])[0].get("words", [])
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

    def get_time(item, key, default):
        val = item.get(key, default)
        try:
            return float(val)
        except Exception:
            return default

    user_words = [w for w in user_words if w.get("word") or w.get("punctuated_word")]
    ai_words = [w for w in ai_words if w.get("word") or w.get("punctuated_word")]

    max_time = 0.0
    for w in user_words + ai_words:
        max_time = max(max_time, get_time(w, "end", 0.0))

    if max_time <= 0:
        return rows

    num_ticks = int(max_time / tick_duration) + 1
    u_tick_words = [[] for _ in range(num_ticks)]
    a_tick_words = [[] for _ in range(num_ticks)]
    u_active_ticks = [False] * num_ticks
    a_active_ticks = [False] * num_ticks

    # Fast O(N) pre-binning of words into ticks
    for w in user_words:
        w_text = (w.get("word") or w.get("punctuated_word") or "").strip()
        start = get_time(w, "start", 0.0)
        end = get_time(w, "end", 0.0)
        mid = (start + end) / 2.0
        mid_idx = int(mid / tick_duration)
        if 0 <= mid_idx < num_ticks and w_text:
            u_tick_words[mid_idx].append(w_text)
        
        s_idx = max(0, int(start / tick_duration))
        e_idx = min(num_ticks, int(end / tick_duration) + 1)
        for i in range(s_idx, e_idx):
            u_active_ticks[i] = True

    for w in ai_words:
        w_text = (w.get("word") or w.get("punctuated_word") or "").strip()
        start = get_time(w, "start", 0.0)
        end = get_time(w, "end", 0.0)
        mid = (start + end) / 2.0
        mid_idx = int(mid / tick_duration)
        if 0 <= mid_idx < num_ticks and w_text:
            a_tick_words[mid_idx].append(w_text)

        s_idx = max(0, int(start / tick_duration))
        e_idx = min(num_ticks, int(end / tick_duration) + 1)
        for i in range(s_idx, e_idx):
            a_active_ticks[i] = True

    user_history = []
    ai_buffer = []

    for idx in range(num_ticks):
        u_tick = u_tick_words[idx]
        a_tick = a_tick_words[idx]
        u_active = u_active_ticks[idx]
        a_active = a_active_ticks[idx]

        if a_active:
            ai_str = "SPEAKING (AI: " + " ".join(ai_buffer[-10:]) + ")" if ai_buffer else "SPEAKING"
        else:
            ai_str = "IDLE"

        hist_str = "User: " + " ".join(user_history[-max_history_words:]) if user_history else ""
        slice_str = " ".join(u_tick) if u_tick else "<silence>"

        # Determine label
        # Case 1: AI is speaking and User cuts in
        if a_active and u_active and u_tick:
            is_backchannel = all(w.lower().strip(".,?!") in BACKCHANNEL_WORDS for w in u_tick)
            label = 2 if is_backchannel else 3
        # Case 2: AI is idle, user just finished speaking in previous tick and now silence
        elif not a_active and not u_active and user_history:
            label = 1
        elif not a_active and u_active:
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

    return rows


def find_all_files(root_dir: str) -> List[str]:
    found = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            found.append(os.path.join(root, f))
    return found


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
        print("Run: uv run python download_hf_datasets.py --repo mrfakename/open-yap-1k")
        return

    all_files = find_all_files(conv_dir)
    print(f"\nScanned '{conv_dir}': found {len(all_files):,} total files.")
    print("Sample discovered files (first 8):")
    for f in all_files[:8]:
        rel = os.path.relpath(f, conv_dir)
        print(f"  - {rel}")

    json_files = [f for f in all_files if f.endswith(".json") or f.endswith(".jsonl")]
    print(f"\nDiscovered {len(json_files):,} JSON/JSONL transcript files.")

    all_rows = []
    processed_convs = 0

    # Strategy 1: Look for combined multichannel or diarized JSON files
    for jf in json_files:
        if processed_convs >= args.max_conversations:
            break
        with open(jf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                words_a, words_b = extract_words_from_deepgram_or_raw(data)
                if words_a and words_b:
                    r1 = process_dual_track_conv(user_words=words_b, ai_words=words_a)
                    r2 = process_dual_track_conv(user_words=words_a, ai_words=words_b)
                    all_rows.extend(r1)
                    all_rows.extend(r2)
                    processed_convs += 1
                    print(f"  Processed multi-track file: {os.path.basename(jf)} (+{len(r1)+len(r2):,} ticks)")
            except Exception:
                pass

    # Strategy 2: If no combined files, pair up separate files (e.g. *_a.json & *_b.json, or by folder)
    if processed_convs == 0:
        dir_groups = {}
        for jf in json_files:
            d = os.path.dirname(jf)
            dir_groups.setdefault(d, []).append(jf)

        for d, files in dir_groups.items():
            if processed_convs >= args.max_conversations:
                break
            files.sort()
            if len(files) >= 2:
                words_a = load_single_file_words(files[0])
                words_b = load_single_file_words(files[1])
                if words_a and words_b:
                    r1 = process_dual_track_conv(user_words=words_b, ai_words=words_a)
                    r2 = process_dual_track_conv(user_words=words_a, ai_words=words_b)
                    all_rows.extend(r1)
                    all_rows.extend(r2)
                    processed_convs += 1
                    print(f"  Paired files in {os.path.basename(d)}: {os.path.basename(files[0])} & {os.path.basename(files[1])} (+{len(r1)+len(r2):,} ticks)")

    print(f"\n============================================================")
    print(f"Converted {processed_convs} dual-perspective conversations into {len(all_rows):,} ticks.")
    print(f"============================================================")

    if all_rows:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        with open(args.output_file, "w", encoding="utf-8") as f:
            for r in all_rows:
                counts[r["label"]] += 1
                fout_line = json.dumps(r)
                f.write(fout_line + "\n")
        print(f"Saved to: {args.output_file}")
        for lbl in range(4):
            pct = (counts[lbl] / len(all_rows) * 100) if all_rows else 0
            print(f"  Class {lbl} ({LABEL_NAMES[lbl]:<10}): {counts[lbl]:,} ({pct:.2f}%)")


if __name__ == "__main__":
    main()

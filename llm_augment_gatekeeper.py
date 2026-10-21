"""
llm_augment_gatekeeper.py

Uses OpenRouter (e.g. deepseek/deepseek-v4.1-flash or deepseek/deepseek-chat)
with MULTITHREADING (concurrent workers) to rapidly synthesize high-diversity
full-duplex conversational training examples:
1. Natural Barge-In Cut-Ins (INTERRUPT - Class 3) over AI speech.
2. Thinking Pauses with <silence> (LISTEN - Class 0 Hard Negatives).
3. Conversational Backchannels (FILLER - Class 2).
4. Clean Turn Completions with <silence> (SPEAK - Class 1).

Features:
- Multithreaded execution (5-8 parallel workers) for 10x faster generation.
- Append-only and fully resumable (stopping with Ctrl+C preserves all generated data).
- Real-time token tracking and strict spending cap ($0.50 budget protection).

Usage:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    uv run python llm_augment_gatekeeper.py --workers 6 --batch-size 8 --target-count 4000
"""

import os
import sys
import json
import time
import argparse
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Pricing per 1M tokens for DeepSeek on OpenRouter ($0.049 / $0.375)
DEFAULT_PROMPT_PRICE_PER_M = 0.049
DEFAULT_COMPLETION_PRICE_PER_M = 0.375

SYSTEM_PROMPT = """You are a conversational linguistics training-data API for a real-time full-duplex voice agent.
Output MUST be ONLY a raw JSON array of objects.
Do NOT include any preamble, thinking process, markdown formatting, or commentary.
Start your response immediately with the character [ and end with ]."""

PROMPT_BARGE_IN = """Generate exactly {batch_size} diverse BARGE-IN interruptions where a user cuts in while the AI is speaking.
Each item must be an object with "text" and "label": 3.
Format of "text": "AI_State: SPEAKING (AI: <ongoing words>) | History: User: <context> | Slice: <cut-in phrase>"

Cover diverse cut-ins: "Wait", "Hold on", "Stop", "Hang on a second", "Wait, what?", "No wait", "No actually", "Wait, which one?", "Excuse me", "Quick question".

Example:
[
  {{"text": "AI_State: SPEAKING (AI: To reset your router, you will need to hold the button for) | History: User: My internet keeps dropping. | Slice: Wait, which button?", "label": 3}},
  {{"text": "AI_State: SPEAKING (AI: The total cost with standard shipping comes out to forty) | History: User: How much for express shipping? | Slice: No, express please", "label": 3}}
]

Output ONLY the JSON array:"""

PROMPT_THINKING_PAUSE = """Generate exactly {batch_size} diverse THINKING PAUSE hard-negatives.
The user is speaking but pauses mid-thought. The mic hears `<silence>`, but the sentence is grammatically incomplete.
Label MUST be 0 (LISTEN).
Format: "AI_State: IDLE | History: User: <incomplete clause> | Slice: <silence>"

Cover trailing conjunctions ("because...", "and then we..."), prepositions ("looking into the...", "decided to..."), and thinking fillers ("so basically, um,...").

Example:
[
  {{"text": "AI_State: IDLE | History: User: I called customer support yesterday because they told me that | Slice: <silence>", "label": 0}},
  {{"text": "AI_State: IDLE | History: User: We could either reschedule for next Tuesday or | Slice: <silence>", "label": 0}}
]

Output ONLY the JSON array:"""

PROMPT_TURN_COMPLETION = """Generate exactly {batch_size} diverse TURN COMPLETIONS.
The user clearly finished asking a question or making a statement, followed by `<silence>`.
Label MUST be 1 (SPEAK).
Format: "AI_State: IDLE | History: User: <complete question or statement> | Slice: <silence>"

Example:
[
  {{"text": "AI_State: IDLE | History: User: Can you tell me if the pharmacy is open on Sundays? | Slice: <silence>", "label": 1}},
  {{"text": "AI_State: IDLE | History: User: Yeah, that will be all for now, thanks. | Slice: <silence>", "label": 1}}
]

Output ONLY the JSON array:"""

PROMPT_BACKCHANNEL = """Generate exactly {batch_size} diverse BACKCHANNEL nods.
The user is talking, and says acknowledgment nods ("yeah", "uh-huh", "right", "got it", "mm-hm", "okay").
Label MUST be 2 (FILLER).
Format: "AI_State: IDLE | History: User: <ongoing user story> | Slice: <nod word>"

Example:
[
  {{"text": "AI_State: IDLE | History: User: So we drove up to the mountains and it started snowing really hard and | Slice: yeah", "label": 2}},
  {{"text": "AI_State: IDLE | History: User: You have to make sure the valve is turned completely clockwise before | Slice: mm-hm", "label": 2}}
]

Output ONLY the JSON array:"""

LABEL_STR_MAP = {
    "LISTEN": 0, "0": 0, 0: 0,
    "SPEAK": 1, "1": 1, 1: 1,
    "FILLER": 2, "2": 2, 2: 2,
    "INTERRUPT": 3, "3": 3, 3: 3
}


def call_openrouter(api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/VoiceFullDuplex",
        "X-Title": "Gatekeeper LLM Augmenter"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2500
    }

    req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=35) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {err_msg}")


def parse_llm_json(raw_text: str) -> List[Dict[str, Any]]:
    if not raw_text or not isinstance(raw_text, str):
        return []

    # 1. Try finding bracketed array [...]
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = raw_text[start:end+1].strip()
        try:
            res = json.loads(snippet)
            if isinstance(res, list):
                return res
        except Exception:
            pass

    # 2. Try parsing individual {...} objects using regex
    import re
    items = []
    for match in re.finditer(r'\{[^{}]+\}', raw_text):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                items.append(obj)
        except Exception:
            pass
    if items:
        return items

    # 3. Fallback direct JSON loads
    clean = raw_text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass

    return []


class ThreadSafeTracker:
    def __init__(self, fout, max_spend, prompt_price, completion_price):
        self.fout = fout
        self.max_spend = max_spend
        self.prompt_price = prompt_price
        self.completion_price = completion_price
        self.lock = threading.Lock()
        self.total_spent = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.generated_counts = {}

    def is_capped(self) -> bool:
        with self.lock:
            return self.total_spent >= self.max_spend

    def record_usage_and_save(self, task_name, usage, items, default_label):
        p_tokens = usage.get("prompt_tokens", 0)
        c_tokens = usage.get("completion_tokens", 0)
        cost = (p_tokens * self.prompt_price / 1_000_000) + (c_tokens * self.completion_price / 1_000_000)

        with self.lock:
            self.total_prompt_tokens += p_tokens
            self.total_completion_tokens += c_tokens
            self.total_spent += cost

            valid_count = 0
            for item in items:
                if not isinstance(item, dict):
                    continue

                text = item.get("text") or item.get("prompt") or item.get("input") or ""
                if not text:
                    ai_state = item.get("ai_state") or item.get("AI_State") or item.get("state") or ("IDLE" if default_label != 3 else "SPEAKING")
                    history = item.get("history") or item.get("History") or ""
                    slice_text = item.get("slice") or item.get("Slice") or "<silence>"
                    if history and not history.startswith("User:"):
                        history = f"User: {history}"
                    text = f"AI_State: {ai_state} | History: {history} | Slice: {slice_text}"

                lbl = item.get("label")
                if lbl is None:
                    lbl = item.get("Label")
                if lbl is not None:
                    final_lbl = LABEL_STR_MAP.get(str(lbl).upper(), default_label)
                else:
                    final_lbl = default_label

                if text.strip() and 0 <= final_lbl <= 3:
                    self.fout.write(json.dumps({"text": text.strip(), "label": final_lbl}) + "\n")
                    valid_count += 1

            self.fout.flush()
            self.generated_counts[task_name] = self.generated_counts.get(task_name, 0) + valid_count

            return valid_count, self.generated_counts[task_name], self.total_spent


def worker_task(api_key, model, task_name, template, default_label, batch_size, tracker):
    if tracker.is_capped():
        return 0

    prompt = template.format(batch_size=batch_size)
    try:
        resp = call_openrouter(api_key, model, prompt)
    except Exception as e:
        return 0

    usage = resp.get("usage", {})
    choices = resp.get("choices", [])
    if not choices:
        return 0

    msg = choices[0].get("message", {})
    content = msg.get("content") or msg.get("reasoning") or ""
    if not content:
        return 0

    items = parse_llm_json(content)
    valid_added, task_total, total_spent = tracker.record_usage_and_save(task_name, usage, items, default_label)
    if valid_added > 0:
        print(f"  [{task_name}] +{valid_added:2d} samples | Task Total: {task_total:,} | Total Spent: ${total_spent:.4f}")
    return valid_added


def main():
    parser = argparse.ArgumentParser(description="Multithreaded OpenRouter LLM Gatekeeper Augmenter.")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""),
                        help="OpenRouter API Key (or export OPENROUTER_API_KEY)")
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash",
                        help="OpenRouter model ID (e.g. deepseek/deepseek-v4.1-flash or deepseek/deepseek-chat)")
    parser.add_argument("--output-file", default="data/gatekeeper_llm_augmented.jsonl")
    parser.add_argument("--target-count", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Samples per request (recommended: 8-10 to avoid token cutoff)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of concurrent worker threads (recommended: 5-8)")
    parser.add_argument("--max-spend", type=float, default=0.50,
                        help="Hard spending cap in USD (default: $0.50)")
    parser.add_argument("--prompt-price", type=float, default=DEFAULT_PROMPT_PRICE_PER_M)
    parser.add_argument("--completion-price", type=float, default=DEFAULT_COMPLETION_PRICE_PER_M)
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: OpenRouter API key not provided!")
        print("Set it via: export OPENROUTER_API_KEY='sk-or-v1-...'")
        sys.exit(1)

    print("=" * 65)
    print("FAST MULTITHREADED GATEKEEPER LLM AUGMENTER")
    print("=" * 65)
    print(f"Model:            {args.model}")
    print(f"Target Samples:   {args.target_count:,}")
    print(f"Concurrent Threads: {args.workers}")
    print(f"Batch Size:       {args.batch_size} samples / request")
    print(f"Max Spending Cap: ${args.max_spend:.2f} USD")
    print(f"Output File:      {args.output_file}")
    print("=" * 65)

    # Count existing samples
    existing_count = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    existing_count += 1
        print(f"[RESUME] Found {existing_count:,} existing samples in {args.output_file}. Resuming safely...")

    needed = max(0, args.target_count - existing_count)
    if needed <= 0:
        print(f"Target count of {args.target_count:,} already reached in {args.output_file}!")
        return

    # Task allocation
    tasks = [
        ("BARGE_IN", PROMPT_BARGE_IN, int(needed * 0.40), 3),
        ("THINKING_PAUSE", PROMPT_THINKING_PAUSE, int(needed * 0.30), 0),
        ("TURN_COMPLETION", PROMPT_TURN_COMPLETION, int(needed * 0.15), 1),
        ("BACKCHANNEL", PROMPT_BACKCHANNEL, int(needed * 0.15), 2),
    ]

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    fout = open(args.output_file, "a", encoding="utf-8")

    tracker = ThreadSafeTracker(fout, args.max_spend, args.prompt_price, args.completion_price)

    try:
        for task_name, template, task_target, default_label in tasks:
            print(f"\n>>> Starting Task: {task_name} (Target: {task_target:,} new samples)")
            task_generated = 0

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                while task_generated < task_target and not tracker.is_capped():
                    futures = []
                    # Submit a wave of concurrent worker requests
                    for _ in range(args.workers):
                        if task_generated >= task_target or tracker.is_capped():
                            break
                        f = executor.submit(
                            worker_task,
                            args.api_key,
                            args.model,
                            task_name,
                            template,
                            default_label,
                            args.batch_size,
                            tracker
                        )
                        futures.append(f)

                    for f in as_completed(futures):
                        added = f.result()
                        task_generated += added

                    time.sleep(0.1)

            if tracker.is_capped():
                print(f"\n[SPEND CAP REACHED] Spending cap ${args.max_spend:.2f} reached. Stopping safely.")
                break

    except KeyboardInterrupt:
        print("\n[STOPPED BY USER] Safely closing file. All generated samples are saved.")
    finally:
        fout.close()

    total_now = existing_count + sum(tracker.generated_counts.values())
    print("\n" + "=" * 65)
    print("RUN COMPLETE")
    print(f"Total Samples Now in File: {total_now:,}")
    print(f"Total Tokens Used:         {tracker.total_prompt_tokens + tracker.total_completion_tokens:,}")
    print(f"Total Cost Spent:          ${tracker.total_spent:.4f} USD")
    print(f"Output File:               {args.output_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()

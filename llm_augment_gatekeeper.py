"""
llm_augment_gatekeeper.py

Uses OpenRouter (e.g. deepseek/deepseek-v4.1-flash-1017 or deepseek/deepseek-chat)
to synthesize high-diversity, human-like full-duplex training examples:
1. Natural Barge-In Cut-Ins (INTERRUPT - Class 3) over AI speech.
2. Thinking Pauses with <silence> (LISTEN - Class 0 Hard Negatives).
3. Conversational Backchannels (FILLER - Class 2).
4. Clean Turn Completions with <silence> (SPEAK - Class 1).

Includes real-time token tracking, live cost calculations, and a strict spending cap
to safeguard your OpenRouter budget (e.g. generates 5,000 samples for under $0.15).

Usage:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    uv run python llm_augment_gatekeeper.py --target-count 4000 --max-spend 0.50
"""

import os
import sys
import json
import time
import random
import argparse
import urllib.request
import urllib.error
from typing import List, Dict, Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Pricing per 1M tokens for DeepSeek on OpenRouter ($0.049 / $0.375)
DEFAULT_PROMPT_PRICE_PER_M = 0.049
DEFAULT_COMPLETION_PRICE_PER_M = 0.375

SYSTEM_PROMPT = """You are a conversational linguistics expert creating training data for an acoustic/linguistic full-duplex conversational voice agent Gatekeeper.
The Gatekeeper receives 600ms ticks formatted as:
`AI_State: <IDLE or SPEAKING (AI: ...)> | History: User: <recent words> | Slice: <what the mic heard in the last 600ms>`

Your task is to generate realistic, diverse, conversational examples in valid JSON format.
Always output ONLY a valid JSON array of objects with the keys "text" and "label". Do not include Markdown fences or extra commentary.
"""

PROMPT_BARGE_IN = """Generate {batch_size} diverse, realistic BARGE-IN interruptions where a user cuts in while the AI is actively speaking.
Format:
- AI_State: SPEAKING (AI: <natural assistant response in progress, 4-15 words>)
- History: User: <previous question or context> (optional or brief)
- Slice: <what the user blurted out in 1-3 words to interrupt>
- Label: 3 (INTERRUPT)

Cover realistic human cut-in variations:
1. Direct stops: "Wait", "Stop", "Hold on", "Hang on a second", "Wait wait wait"
2. Corrections: "No wait", "No actually", "Wait, that's wrong", "Not that one"
3. Questions/clarifications: "Quick question", "Wait, what?", "Excuse me", "Can I ask something?", "Wait why?"
4. Disagreements: "No no no", "I disagree", "Wait don't do that"

Example output:
[
  {{"text": "AI_State: SPEAKING (AI: To reset your router, you will need to hold the button for) | History: User: My internet keeps dropping. | Slice: Wait, which button?", "label": 3}},
  {{"text": "AI_State: SPEAKING (AI: The total cost with standard shipping comes out to forty) | History: User: How much for express shipping? | Slice: No, express please", "label": 3}}
]

Output ONLY the JSON array of {batch_size} items:"""

PROMPT_THINKING_PAUSE = """Generate {batch_size} diverse, realistic THINKING PAUSE hard-negatives.
The user is speaking but pauses in the middle of a thought. The mic hears `<silence>`, but the sentence is grammatically or semantically INCOMPLETE.
The AI MUST NOT interrupt or speak. The true label MUST BE 0 (LISTEN).

Format:
- AI_State: IDLE
- History: User: <user speech that trailingly cuts off mid-thought>
- Slice: <silence>
- Label: 0 (LISTEN)

Cover realistic hesitations:
1. Trailing conjunctions: "...because I thought that", "...and then maybe we", "...or if you could"
2. Trailing prepositions: "...looking into the", "...decided to go to", "...with the new"
3. Thinking fillers: "...so basically, um,", "...let me see if", "...well honestly,"
4. False starts: "...wait let me check", "...I was wondering whether or not"

Example output:
[
  {{"text": "AI_State: IDLE | History: User: I called customer support yesterday because they told me that | Slice: <silence>", "label": 0}},
  {{"text": "AI_State: IDLE | History: User: We could either reschedule for next Tuesday or | Slice: <silence>", "label": 0}}
]

Output ONLY the JSON array of {batch_size} items:"""

PROMPT_TURN_COMPLETION = """Generate {batch_size} diverse, natural TURN COMPLETIONS.
The user has clearly finished asking a question, making a request, or completing a thought, followed by `<silence>`.
The AI SHOULD TAKE THE TURN. The label MUST BE 1 (SPEAK).

Format:
- AI_State: IDLE
- History: User: <clear, complete user utterance>
- Slice: <silence>
- Label: 1 (SPEAK)

Include diverse questions, confirmations, statements, and short answers across tech, customer service, booking, casual chat, and troubleshooting.

Example output:
[
  {{"text": "AI_State: IDLE | History: User: Can you tell me if the pharmacy is open on Sundays? | Slice: <silence>", "label": 1}},
  {{"text": "AI_State: IDLE | History: User: Yeah, that will be all for now, thanks. | Slice: <silence>", "label": 1}}
]

Output ONLY the JSON array of {batch_size} items:"""

PROMPT_BACKCHANNEL = """Generate {batch_size} diverse BACKCHANNEL occurrences.
The user is speaking or narrating, and says conversational nods or acknowledgment fillers ("yeah", "uh-huh", "right", "got it", "mm-hm", "okay").
The label MUST BE 2 (FILLER).

Format:
- AI_State: IDLE
- History: User: <ongoing user clause or story>
- Slice: <backchannel word>
- Label: 2 (FILLER)

Example output:
[
  {{"text": "AI_State: IDLE | History: User: So we drove up to the mountains and it started snowing really hard and | Slice: yeah", "label": 2}},
  {{"text": "AI_State: IDLE | History: User: You have to make sure the valve is turned completely clockwise before | Slice: mm-hm", "label": 2}}
]

Output ONLY the JSON array of {batch_size} items:"""


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
        "temperature": 0.85,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"} if "flash" in model else None
    }

    req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {err_msg}")


LABEL_STR_MAP = {
    "LISTEN": 0, "0": 0, 0: 0,
    "SPEAK": 1, "1": 1, 1: 1,
    "FILLER": 2, "2": 2, 2: 2,
    "INTERRUPT": 3, "3": 3, 3: 3
}


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


def main():
    parser = argparse.ArgumentParser(description="Generate augmented training data via OpenRouter LLM.")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""),
                        help="OpenRouter API Key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017",
                        help="OpenRouter model ID (e.g. deepseek/deepseek-v4.1-flash-1017 or deepseek/deepseek-chat)")
    parser.add_argument("--output-file", default="data/gatekeeper_llm_augmented.jsonl")
    parser.add_argument("--target-count", type=int, default=4000,
                        help="Total number of augmented samples to generate across classes")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of samples to generate per LLM call (recommended: 15-25)")
    parser.add_argument("--max-spend", type=float, default=0.50,
                        help="Safety spending cap in USD (default: $0.50)")
    parser.add_argument("--prompt-price", type=float, default=DEFAULT_PROMPT_PRICE_PER_M)
    parser.add_argument("--completion-price", type=float, default=DEFAULT_COMPLETION_PRICE_PER_M)
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: OpenRouter API key not provided!")
        print("Set it via export OPENROUTER_API_KEY='sk-or-v1-...' or pass --api-key.")
        sys.exit(1)

    print("=" * 65)
    print("GATEKEEPER LLM DATASET AUGMENTER (OpenRouter DeepSeek)")
    print("=" * 65)
    print(f"Model:            {args.model}")
    print(f"Target Samples:   {args.target_count:,}")
    print(f"Batch Size:       {args.batch_size} per request")
    print(f"Max Spend Cap:    ${args.max_spend:.2f} USD")
    print(f"Pricing:          ${args.prompt_price:.3f} / 1M prompt | ${args.completion_price:.3f} / 1M completion")
    print(f"Output:           {args.output_file}")
    print("=" * 65)

    # Check existing count
    existing_count = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    existing_count += 1
        print(f"Found existing file with {existing_count:,} samples. Resuming generation...")

    needed = max(0, args.target_count - existing_count)
    if needed <= 0:
        print(f"Target count of {args.target_count:,} already reached in {args.output_file}!")
        return

    # Task allocation: 40% Barge-In (Class 3), 30% Thinking Pause (Class 0), 15% SPEAK (Class 1), 15% FILLER (Class 2)
    tasks = [
        ("BARGE_IN", PROMPT_BARGE_IN, int(needed * 0.40), 3),
        ("THINKING_PAUSE", PROMPT_THINKING_PAUSE, int(needed * 0.30), 0),
        ("TURN_COMPLETION", PROMPT_TURN_COMPLETION, int(needed * 0.15), 1),
        ("BACKCHANNEL", PROMPT_BACKCHANNEL, int(needed * 0.15), 2),
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_spent = 0.0
    generated_this_run = 0

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    fout = open(args.output_file, "a", encoding="utf-8")

    try:
        for task_name, template, task_target, default_label in tasks:
            print(f"\n>>> Starting Task: {task_name} (Goal: {task_target:,} samples)")
            task_generated = 0

            while task_generated < task_target:
                # Check spend cap
                if total_spent >= args.max_spend:
                    print(f"\n[SPEND CAP REACHED] Total spent (${total_spent:.4f}) reached max cap (${args.max_spend:.2f}). Stopping.")
                    break

                batch_n = min(args.batch_size, task_target - task_generated)
                formatted_prompt = template.format(batch_size=batch_n)

                try:
                    resp = call_openrouter(args.api_key, args.model, formatted_prompt)
                except Exception as e:
                    print(f"  Warning: API call failed: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

                # Token & cost calculation
                usage = resp.get("usage", {})
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)
                total_prompt_tokens += p_tokens
                total_completion_tokens += c_tokens

                cost_this_call = (p_tokens * args.prompt_price / 1_000_000) + (c_tokens * args.completion_price / 1_000_000)
                total_spent += cost_this_call

                # Parse choices
                choices = resp.get("choices", [])
                if not choices:
                    print(f"  Warning: No choices returned in response: {resp}")
                    time.sleep(2)
                    continue

                msg = choices[0].get("message", {})
                content = msg.get("content") or msg.get("reasoning") or ""
                if not content:
                    print(f"  Warning: Received empty content from API choice: {choices[0]}")
                    time.sleep(2)
                    continue

                items = parse_llm_json(content)

                valid_items = 0
                for item in items:
                    if not isinstance(item, dict):
                        continue

                    # Extract or construct text
                    text = item.get("text") or item.get("prompt") or item.get("input") or ""
                    if not text:
                        ai_state = item.get("ai_state") or item.get("AI_State") or item.get("state") or ("IDLE" if default_label != 3 else "SPEAKING")
                        history = item.get("history") or item.get("History") or ""
                        slice_text = item.get("slice") or item.get("Slice") or "<silence>"
                        if history and not history.startswith("User:"):
                            history = f"User: {history}"
                        text = f"AI_State: {ai_state} | History: {history} | Slice: {slice_text}"

                    # Extract or infer label
                    lbl = item.get("label")
                    if lbl is None:
                        lbl = item.get("Label")
                    if lbl is not None:
                        final_lbl = LABEL_STR_MAP.get(str(lbl).upper(), default_label)
                    else:
                        final_lbl = default_label

                    if text.strip() and 0 <= final_lbl <= 3:
                        fout.write(json.dumps({"text": text.strip(), "label": final_lbl}) + "\n")
                        valid_items += 1

                if valid_items == 0:
                    print(f"  [Debug] 0 items parsed from batch. Model returned:\n    {repr(content[:300])}")
                    time.sleep(1)

                fout.flush()
                task_generated += valid_items
                generated_this_run += valid_items

                print(f"  [{task_name}] +{valid_items} samples | Progress: {task_generated:,}/{task_target:,} | Spent so far: ${total_spent:.4f}")
                time.sleep(0.3)  # Brief rate-limit buffer

            if total_spent >= args.max_spend:
                break

    finally:
        fout.close()

    print("\n" + "=" * 65)
    print("GENERATION RUN COMPLETE")
    print("=" * 65)
    print(f"Samples Generated this run: {generated_this_run:,}")
    print(f"Total File Count:           {existing_count + generated_this_run:,}")
    print(f"Total Tokens Used:          {total_prompt_tokens + total_completion_tokens:,} ({total_prompt_tokens:,} prompt, {total_completion_tokens:,} completion)")
    print(f"Total Cost:                 ${total_spent:.4f} USD (Budget remaining: ${(args.max_spend - total_spent):.4f})")
    print(f"Saved to:                   {args.output_file}")


if __name__ == "__main__":
    main()


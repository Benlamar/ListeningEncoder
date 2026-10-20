"""
test_gatekeeper.py

Interactive CLI and Benchmark Validator for the trained Gatekeeper.
Tests both interactive user typing (Enter = <silence>) and standard duplex scenarios
(Hesitation, Clean Turn Completion, Barge-In, and Backchannel).

Usage:
    uv run python test_gatekeeper.py --checkpoint checkpoints/gatekeeper_model/final_best
    uv run python test_gatekeeper.py --benchmark --checkpoint checkpoints/gatekeeper_model/final_best
"""

import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_NAMES = {0: "LISTEN", 1: "SPEAK", 2: "FILLER", 3: "INTERRUPT"}

BENCHMARK_SCENARIOS = [
    {
        "name": "Hesitation Pause (Thinking)",
        "state": "IDLE",
        "history": "User: I was thinking about learning piano because",
        "slice": "<silence>",
        "expected": "LISTEN (0)"
    },
    {
        "name": "Clean Turn Completion (Question)",
        "state": "IDLE",
        "history": "User: Can you tell me what time the store opens?",
        "slice": "<silence>",
        "expected": "SPEAK (1)"
    },
    {
        "name": "Mid-Sentence Clause",
        "state": "IDLE",
        "history": "User: Yesterday I drove up the coast",
        "slice": "and the traffic was crazy",
        "expected": "LISTEN (0)"
    },
    {
        "name": "Barge-in while AI speaks",
        "state": "SPEAKING (AI: Modern chargers usually stop automatically...)",
        "history": "User: Can I leave it overnight?",
        "slice": "Wait, hold on a second!",
        "expected": "INTERRUPT (3)"
    },
    {
        "name": "Active Listening Backchannel Opportunity",
        "state": "IDLE",
        "history": "User: So I went to the dealership and the manager was talking to another customer for twenty minutes and then",
        "slice": "finally came over",
        "expected": "FILLER (2) or LISTEN (0)"
    }
]


def predict(text: str, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=160).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    
    pred_class = int(probs.argmax())
    return pred_class, probs


def run_benchmarks(model, tokenizer, device):
    print("\n" + "="*70)
    print("RUNNING GATEKEEPER BENCHMARK VALIDATION SCENARIOS")
    print("="*70)
    
    for idx, sc in enumerate(BENCHMARK_SCENARIOS, 1):
        prompt = f"AI_State: {sc['state']} | History: {sc['history']} | Slice: {sc['slice']}"
        pred_class, probs = predict(prompt, model, tokenizer, device)
        pred_name = LABEL_NAMES[pred_class]
        
        prob_str = " | ".join([f"{LABEL_NAMES[i]}: {probs[i]:.2f}" for i in range(4)])
        print(f"\nScenario {idx}: {sc['name']}")
        print(f"  Input:    {prompt}")
        print(f"  Expected: {sc['expected']}")
        print(f"  Result:   {pred_name} (Class {pred_class})")
        print(f"  Probabilities: [{prob_str}]")


def interactive_mode(model, tokenizer, device):
    print("\n" + "="*70)
    print("INTERACTIVE TICK CONSOLE (Press Enter for <silence>, Ctrl+C to exit)")
    print("Type 1-3 words representing what the mic heard in the last 600 ms.")
    print("="*70)

    ai_state = "IDLE"
    history = []

    while True:
        try:
            line = input("\n[Mic heard]> ").strip()
            slice_text = "<silence>" if not line else line

            prompt = f"AI_State: {ai_state} | History: {' '.join(history[-30:])} | Slice: {slice_text}"
            pred_class, probs = predict(prompt, model, tokenizer, device)
            pred_name = LABEL_NAMES[pred_class]

            prob_str = " | ".join([f"{LABEL_NAMES[i]}: {probs[i]:.2f}" for i in range(4)])
            print(f" -> Decision: {pred_name:<10} (probs: {prob_str})")

            # Update mock state
            if line:
                history.append(line)
            if pred_class == 1: # SPEAK
                print("  [System]: Dispatched turn to LLM! AI transitions to SPEAKING.")
                ai_state = "SPEAKING"
            elif pred_class == 3: # INTERRUPT
                print("  [System]: INTERRUPT DETECTED! Aborting active speech.")
                ai_state = "IDLE"

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive test.")
            break


def main():
    parser = argparse.ArgumentParser(description="Test trained gatekeeper model.")
    parser.add_argument("--checkpoint", default="answerdotai/ModernBERT-base",
                        help="Path to trained checkpoint or HF model ID")
    parser.add_argument("--benchmark", action="store_true", help="Run automated benchmark scenarios")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint '{args.checkpoint}' on {device.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=4).to(device)
    model.eval()

    if args.benchmark:
        run_benchmarks(model, tokenizer, device)
    else:
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()

# Master Execution Plan: Real-Time Micro-Turn Duplex AI Engine

**Hardware target:** NVIDIA DGX Spark (128 GB unified coherent memory)  
**Scope:** Text-in/text-out, 600 ms micro-turn conversational engine. STT/TTS are bypassed during core model validation.  
**Base models:**

- **Tier 1 — Gatekeeper:** `answerdotai/ModernBERT-base` (~150M parameters, encoder)
- **Tier 2 — Speaker:** `google/gemma-2-9b-it` or `google/gemma-3-12b-it` (decoder served with vLLM)

---

## Table of Contents

1. [Finalized System Architecture](#1-finalized-system-architecture)
2. [Phase 1: Environment and Smoke-Test Setup](#2-phase-1-environment-and-smoke-test-setup)
3. [Phase 2: Gatekeeper Dataset Engineering](#3-phase-2-gatekeeper-dataset-engineering)
4. [Phase 3: Train and Test Tier 1](#4-phase-3-train-and-test-tier-1-modernbert)
5. [Phase 4: Orchestrate Tier 1 with Gemma on vLLM](#5-phase-4-orchestrate-tier-1-with-gemma-on-vllm)
6. [Phase 5: Train a Gemma Duplex LoRA](#6-phase-5-track-2-train-a-gemma-duplex-lora)
7. [Guardrails and Implementation Notes](#7-guardrails-and-implementation-notes)
8. [Next Step](#8-next-step)

---

## 1. Finalized System Architecture

The system is designed to approximate real-time duplex behavior—continuing to listen while the AI is speaking—without waking the large language model for every incoming slice.

### 1.1 Every 600 ms tick

Each tick receives a small incoming text slice, typically one to three words, plus recent conversational history and the current AI state.

```text
Incoming 1–3 word text slice
             |
             v
+----------------------------------------------------------+
| TIER 1: ModernBERT Decision Gatekeeper                   |
| Input: recent turns + AI_State + current text slice      |
| Output: probabilities for four classes                  |
+-------------------------------+--------------------------+
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
       LISTEN (0)          FILLER (2)        INTERRUPT (3)
             |                  |                  |
       Keep collecting     Emit a brief       If AI is speaking,
       user input; do      backchannel such    abort generation
       not call the LLM    as "Mhm" or "Right" and clear unspoken
                                                buffered words

                    SPEAK (1)
                       |
                       v
             Dispatch context to Gemma
             on the vLLM server
                       |
                       v
             Stream generated text into a buffer
                       |
                       v
             Release up to 3 words per 600 ms tick
             while continuing to classify user input
```

### 1.2 Four gatekeeper classes

| Label | Class | Intended behavior |
|---:|---|---|
| `0` | `LISTEN` | Continue collecting user input. Do not start a response yet. |
| `1` | `SPEAK` | Treat the user turn as complete and request a response from Gemma. |
| `2` | `FILLER` | Optionally emit a short backchannel such as “Mhm” or “Right.” |
| `3` | `INTERRUPT` | If the AI is speaking, stop generation/release and discard unspoken buffered text. Safety-related cases may require an immediate warning. |

**Important:** These are initial design targets, not guaranteed model behavior. The dataset, thresholds, and timing must be validated empirically. In particular, a text-only gatekeeper cannot reliably detect speech onset, silence, or acoustic urgency without an upstream speech/activity signal. During this project, those signals are represented by text slices and a `<silence>` marker.

---

## 2. Phase 1: Environment and Smoke-Test Setup

**Goal:** Confirm the environment and model loading before generating a large dataset.

### 2.1 Create a virtual environment

```bash
python3 -m venv duplex_env
source duplex_env/bin/activate
python -m pip install --upgrade pip
pip install torch transformers datasets peft trl accelerate vllm requests scikit-learn
```

Check the installed versions and resolve any compatibility issues between PyTorch, CUDA, Transformers, and vLLM before proceeding. The exact installation commands may differ for the DGX Spark software image.

### 2.2 Verify hardware and precision

Run a small diagnostic:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
```

Do not assume a particular device name or precision support; verify the output on the actual machine.

### 2.3 Download the base models

Download or cache the models before training so that training runs are not delayed by network access:

- `answerdotai/ModernBERT-base`
- `google/gemma-2-9b-it` **or** `google/gemma-3-12b-it`

Gemma access may require accepting the model's license and authenticating with the model host. Confirm that the selected model architecture and version are supported by the installed vLLM release.

### Phase 1 checklist

- [ ] Create and activate `duplex_env`.
- [ ] Confirm PyTorch and GPU visibility.
- [ ] Verify BF16 support.
- [ ] Download both selected models.
- [ ] Confirm that ModernBERT and the chosen Gemma model load successfully.

---

## 3. Phase 2: Gatekeeper Dataset Engineering

This is a high-impact part of the system: ambiguous pauses, incomplete thoughts, interruptions, and genuine turn completion must be represented well.

### 3.1 JSONL schema

File: `gatekeeper_data.jsonl`

Each line is one JSON object containing:

- `text`: the current AI state, recent history, and new text slice
- `label`: integer class (`0`, `1`, `2`, or `3`)

Example rows:

```jsonl
{"text":"AI_State: IDLE | History: User: Hey can you | Slice: help me out?","label":0}
{"text":"AI_State: IDLE | History: User: Hey can you help me out? | Slice: <silence>","label":1}
{"text":"AI_State: IDLE | History: User: I want to learn piano because | Slice: <silence>","label":0}
{"text":"AI_State: IDLE | History: User: Yesterday I drove all the way up the coast | Slice: and the traffic was crazy","label":0}
{"text":"AI_State: IDLE | History: User: I am going to mix this bleach | Slice: with ammonia right now","label":3}
{"text":"AI_State: SPEAKING (AI: Loose gas caps usually...) | History: User: Should I check it? | Slice: <silence>","label":0}
{"text":"AI_State: SPEAKING (AI: Loose gas caps usually...) | History: User: Should I check it? | Slice: Wait, hold on!","label":3}
```

**Labeling note:** The labels above are illustrative and should be reviewed before training. For example, a mid-sentence slice should normally be `LISTEN` (`0`), while a completed thought followed by a reliable end-of-turn signal may be `SPEAK` (`1`). A safety label should reflect the defined annotation policy, not just the presence of a hazardous keyword.

### 3.2 Initial data targets

Start with a small smoke-test dataset. A suggested distribution for a later 2,000-row dataset is:

| Scenario | Target share | Typical label |
|---|---:|---|
| Standard mid-sentence slices | 35% | `0 — LISTEN` |
| Hesitation and thinking pauses | 20% | `0 — LISTEN` |
| Clean turn completion | 25% | `1 — SPEAK` |
| Active-listening/backchannel opportunities | 10% | `2 — FILLER` |
| Barge-in and urgent safety interruptions | 10% | `3 — INTERRUPT` |

These percentages are starting points, not universal optimal values. Because the final two categories contain distinct behaviors, ensure the interruption category includes both user barge-in examples and urgent safety examples. Evaluate performance per class rather than relying on overall accuracy.

#### Scenario A: Standard mid-sentence slices

Examples:

- “I was thinking”
- “about my car”
- “because the last time…”

The gatekeeper should generally keep listening rather than start a response.

#### Scenario B: Hesitation and thinking pauses

Examples:

- “and…” followed by `<silence>`
- “but…” followed by `<silence>`
- “because…” followed by `<silence>`
- “uh…” followed by `<silence>`

The goal is to reduce premature turn completion when the user is still formulating a thought.

#### Scenario C: Clean turn completion

Examples:

- A complete question followed by `<silence>`
- A complete statement followed by an end-of-turn signal

The target is `SPEAK` (`1`) when the available evidence supports that the user has finished.

#### Scenario D: Active listening and backchannels

Examples:

- The user is well into a multi-sentence story.
- A short “Mhm” or “Right” would be appropriate without taking over the conversation.

Use this class sparingly and test whether backchannels feel natural rather than intrusive.

#### Scenario E: Barge-in and safety interruptions

Include both:

1. **User barge-in:** `AI_State` is `SPEAKING`, and the user says “Wait,” “No, I meant…,” or similar.
2. **Urgent hazard:** The user describes an immediate physical danger, such as an imminent attempt to mix bleach and ammonia or touch a live wire.

Safety behavior should not depend exclusively on this classifier. For high-risk cases, use a separate, tested safety policy and avoid making the classifier the sole mechanism for urgent intervention.

---

## 4. Phase 3: Train and Test Tier 1 (ModernBERT)

### 4.1 Start with a smoke test

- [ ] Generate approximately 300 labeled slices from about 50 example conversations.
- [ ] Split the data into training and validation sets by conversation, not by randomly splitting near-duplicate slices. This helps avoid leakage.
- [ ] Fine-tune ModernBERT for three epochs as an initial experiment.
- [ ] Record actual training time; do not assume it will complete in under two minutes on every software configuration.

### 4.2 Evaluate the model

Review:

- Validation accuracy
- Per-class precision, recall, and F1
- Confusion matrix
- False `SPEAK` predictions during hesitations
- False `LISTEN` predictions after genuine turn completion
- Missed interruptions and unnecessary interruptions

Prioritize the consequences of each error. Overall accuracy can hide poor recall for rare but important classes.

### 4.3 Apply confidence thresholds

An initial inference policy to test:

```python
def choose_action(probabilities):
    """
    probabilities: mapping from integer class ID to probability.
    Class IDs:
      0 = LISTEN
      1 = SPEAK
      2 = FILLER
      3 = INTERRUPT
    """
    if probabilities[3] >= 0.50:
        return 3  # INTERRUPT

    if probabilities[1] >= 0.65:
        return 1  # SPEAK

    if probabilities[2] >= 0.60:
        return 2  # FILLER

    return 0      # Default to LISTEN
```

These thresholds are provisional. Tune them on a held-out validation set, and check probability calibration. A threshold is not meaningful unless the model's confidence estimates are reasonably calibrated.

### 4.4 Interactive terminal test

Build a simple command-line loop that accepts two or three words at a time. Press Enter to represent `<silence>`. Display the predicted class and its confidence.

The purpose is to validate the data format, model inference, and turn-taking logic before introducing vLLM.

### Phase 3 exit criteria

- [ ] Training and inference complete without errors.
- [ ] Validation metrics and a confusion matrix are saved.
- [ ] Hesitation examples are not routinely classified as completed turns.
- [ ] Thresholds are documented and justified by validation results.
- [ ] The interactive test works repeatably.

---

## 5. Phase 4: Orchestrate Tier 1 with Gemma on vLLM

### 5.1 Start the vLLM server

Use the selected Gemma model and BF16 where supported by the hardware and model. Enable prefix caching using the flag supported by your installed vLLM version (commonly `--enable-prefix-caching`).

Check the installed version's help output and documentation before choosing command-line options; flags can change across releases.

### 5.2 Build the three-word paced-commit loop

When ModernBERT predicts `SPEAK`:

1. Send the current conversation context to Gemma through vLLM.
2. Stream or collect generated text into an output buffer.
3. Release up to three words per 600 ms tick to the terminal.
4. Continue processing incoming user slices while output is being released.
5. If an interruption is detected, cancel the active generation request when possible.
6. Discard unspoken words remaining in the local buffer.
7. Keep only the text actually committed to the conversation transcript as spoken assistant output.

The three-word cadence is a pacing target for the prototype, not a guarantee of a 600 ms end-to-end response. Measure first-token latency, generation throughput, classifier latency, cancellation latency, and end-to-end turn latency separately.

### 5.3 Prefix-cache considerations

To improve prefix-cache reuse:

- Keep stable system instructions and earlier conversation text unchanged.
- Append new conversation content instead of rewriting the entire history.
- Avoid modifying old prompt text while a request is in flight.
- Keep request formatting and tokenization consistent.
- Measure actual cache-hit behavior; do not assume a 99% hit rate.

Prefix caching helps when prompts share an identical prefix. It does not guarantee that every request will hit the cache, nor does it eliminate generation latency.

### 5.4 Interruption behavior

If the user says “Wait, stop” while Gemma is speaking:

- Stop releasing buffered words immediately.
- Cancel the generation request if supported by the client/server path.
- Preserve already committed assistant text.
- Discard text that was generated but never released.
- Add the user's interruption to the transcript and continue the turn-taking loop.

Test the race condition where additional tokens arrive after cancellation has been requested.

---

## 6. Phase 5: Track 2 — Train a Gemma Duplex LoRA

Start this phase only after the gatekeeper plus stock Gemma prototype works end to end.

### Mode A: Spoken-style LoRA (recommended)

Keep ModernBERT as the gatekeeper and fine-tune Gemma to produce concise, natural spoken responses. Desired behavior includes:

- Short, conversational sentences
- Minimal unnecessary preambles
- No default Markdown lists unless useful
- No formulaic openings such as “Certainly! Here are five tips”
- Responses that are easy to interrupt and resume

This mode improves response style without asking Gemma to make every turn-taking decision.

### Mode B: Single-model experiment

Fine-tune Gemma to emit control decisions such as `WAIT` versus response text, or to experiment with micro-turn output. Compare this design against the two-tier architecture on latency, stability, and conversational quality.

Treat Mode B as an experiment rather than an assumed improvement: generating control tokens through a large decoder can add latency and create more complicated serving behavior.

### 6.1 LoRA checklist

- [ ] Avoid resizing token embeddings unless a new token is genuinely necessary. Prefer an existing token or a clearly defined text marker such as `WAIT`, and verify its tokenization.
- [ ] Mask user prompts and non-target tokens in the loss so training focuses on assistant completions. Confirm that the chosen TRL/data-collator setup actually applies the intended masking.
- [ ] Start with the following hyperparameters and tune from measured results:

```python
r = 16
lora_alpha = 32
lora_dropout = 0.05

target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

- [ ] **Experiment 1:** Adapt all supported target modules.
- [ ] **Experiment 2:** Adapt only the final four transformer layers if supported by the chosen PEFT/model configuration.
- [ ] Compare training cost, response style, interruption behavior, and inference latency.

Do not assume that avoiding embedding resizing guarantees a small adapter: adapter size depends on target modules, rank, layer count, and saved parameters.

### 6.2 Optional `WAIT` logit-bias experiment (Mode B)

If the serving stack exposes the necessary token-level probabilities and supports logit bias, test a rule that suppresses `WAIT` when its probability is below a chosen threshold—for example, 0.30.

Before implementing this:

- Verify that `WAIT` maps to the intended token ID or token sequence.
- Confirm how the server applies logit bias and whether it supports the desired per-request behavior.
- Evaluate on held-out sequences to ensure the rule does not force inappropriate speech or suppress necessary listening.
- Prefer explicit control logic when it is simpler and more reliable than token-probability heuristics.

A hard mask such as `-100` is an aggressive intervention and should not be used without careful validation.

---

## 7. Guardrails and Implementation Notes

| Do | Avoid |
|---|---|
| Start with a 300-row smoke-test dataset. | Generating tens of thousands of rows before running the pipeline once. |
| Append new turns while keeping the existing prompt prefix stable. | Rewriting or summarizing earlier prompt content on every tick without measuring the cache impact. |
| Include examples where `AI_State` is `SPEAKING`. | Training only on turn-by-turn conversations where the user always waits for the AI to finish. |
| Keep ModernBERT's input history bounded (initial target: roughly 150 tokens). | Sending thousands of history tokens to the classifier on every tick. |
| Evaluate per-class metrics and interruption failure cases. | Relying on overall accuracy alone. |
| Measure cache hits and latency in the actual serving setup. | Assuming a 99% prefix-cache hit rate or guaranteed sub-600 ms responses. |
| Cancel generation and discard unspoken buffered output on interruption. | Keeping generated-but-unspoken text in the visible transcript. |
| Use an independent safety policy for urgent hazards. | Relying on a text classifier as the sole safety mechanism. |

### Key engineering caveats

1. **Silence detection:** A text marker such as `<silence>` is only a stand-in during this text-only prototype. A production voice system needs an upstream voice activity/end-of-turn signal.
2. **Latency:** A 600 ms tick interval does not itself guarantee that classification, LLM generation, cancellation, and output pacing all meet that budget.
3. **Prefix caching:** Cache reuse depends on exact shared token prefixes and serving behavior; validate with measurements.
4. **Data quality:** Synthetic examples are useful for smoke tests, but final performance requires diverse, reviewed examples and held-out evaluation.
5. **Safety:** Hazard detection should be independently tested and should not rely solely on a single learned class.

---

## 8. Next Step

Build the first end-to-end smoke-test script that:

1. Generates a small, labeled dataset of sliced conversations.
2. Fine-tunes ModernBERT on the dataset.
3. Reports validation metrics and a confusion matrix.
4. Starts an interactive terminal test for class predictions.

Keep the initial goal narrow: verify the data pipeline and model training before integrating vLLM or beginning LoRA experiments.

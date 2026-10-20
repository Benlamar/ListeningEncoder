# Real-Time Micro-Turn Duplex AI Engine

Two-Tier Full-Duplex conversational voice engine utilizing a low-latency gatekeeper (`answerdotai/ModernBERT-base` or `jhu-clsp/ettin-encoder-150m`) paired with a high-capacity speaker LLM (`google/gemma-2-9b-it` on vLLM).

## Setup & Running with `uv`

```bash
# 1. Install/sync environment dependencies
uv sync

# 2. Prepare dataset
uv run python prepare_gatekeeper_dataset.py

# 3. Train Gatekeeper
uv run python train_gatekeeper.py --model_name answerdotai/ModernBERT-base

# 4. Test interactive CLI & benchmarks
uv run python test_gatekeeper.py --benchmark
```

See [README_RUN_GUIDE.md](README_RUN_GUIDE.md) and [FINAL_DUPLEX_MASTER_PLAN.md](FINAL_DUPLEX_MASTER_PLAN.md) for architectural details and full benchmarks.

"""
resplit_data.py
Run from: /Users/irtiza/PyCharmMiscProject/cognitune/
Usage: python3 resplit_data.py

Combines all batch files, shuffles, and splits into 90/10 train/val.
"""

import json
import random
import os

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_FILES = [
    "data/train.jsonl",          # original ~10 examples
    "data/train_batch2.jsonl",        # 40 examples
    "data/train_batch3.jsonl",        # 50 examples
    "data/train_batch4.jsonl",        # 100 examples
    "data/train_batch5.jsonl",        # 100 examples
    "data/train_batch6.jsonl",        # 100 examples
    "data/train_varied_batch1.jsonl", # 32 examples
    "data/train_varied_batch2.jsonl", # 23 examples
]
OUTPUT_TRAIN = "data/train.jsonl"
OUTPUT_VALID = "data/valid.jsonl"
VAL_RATIO    = 0.10             # 10% val → ~40 examples from 400 total
SEED         = 42
CHAT_TEMPLATE = "<|im_start|>user\n{instruction}\nassistant\n{output}"
# ─────────────────────────────────────────────────────────────────────────────


def load_jsonl(path):

    examples = []
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return examples
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [WARN] Skipping bad line in {path}: {e}")
    return examples


def to_mlx_format(example):
    """Convert Alpaca-style dict to MLX chat format."""
    # Already in MLX format (has 'text' key)
    if "text" in example:
        return example
    # Alpaca format (has 'instruction' and 'output')
    instruction = example.get("instruction", "").strip()
    output      = example.get("output", "").strip()
    inp         = example.get("input", "").strip()
    if inp:
        instruction = f"{instruction}\n{inp}"
    return {"text": CHAT_TEMPLATE.format(instruction=instruction, output=output)}


def main():
    print("=" * 60)
    print("CogniTune — Data Resplit")
    print("=" * 60)

    # 1. Load all examples
    all_examples = []
    for path in BATCH_FILES:
        batch = load_jsonl(path)
        print(f"  Loaded {len(batch):>4} examples from {path}")
        all_examples.extend(batch)

    print(f"\n  Total raw examples: {len(all_examples)}")

    # 2. Convert to MLX format
    all_examples = [to_mlx_format(e) for e in all_examples]

    # 3. Deduplicate by text content
    seen = set()
    unique = []
    for ex in all_examples:
        key = ex["text"][:120]   # first 120 chars as fingerprint
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    print(f"  After dedup:        {len(unique)}")

    # 4. Shuffle and split
    random.seed(SEED)
    random.shuffle(unique)

    val_size   = max(40, int(len(unique) * VAL_RATIO))   # at least 40 val
    val_size   = min(val_size, 60)                        # cap at 60 val
    train_data = unique[val_size:]
    val_data   = unique[:val_size]

    print(f"  Train examples:     {len(train_data)}")
    print(f"  Val   examples:     {len(val_data)}")

    # 5. Write output files
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_TRAIN, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    with open(OUTPUT_VALID, "w") as f:
        for ex in val_data:
            f.write(json.dumps(ex) + "\n")

    print(f"\n  Written → {OUTPUT_TRAIN}")
    print(f"  Written → {OUTPUT_VALID}")
    print("\n  Done. Ready to train.")
    print("=" * 60)


if __name__ == "__main__":
    main()

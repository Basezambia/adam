"""Prepare a small OpenWebText subset (~200MB tokens) for ADAM training.

Streams the dataset to avoid 54GB download, tokenizes with GPT-2 BPE,
writes to train.bin / val.bin as uint16 memmap (compatible with nanoGPT loader).
"""
import os
import sys
import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'owt_small')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_TRAIN_TOKENS = 100_000_000   # 100M train tokens (~200MB uint16)
TARGET_VAL_TOKENS   = 1_000_000     # 1M val tokens

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

def tokenize_stream(dataset_iter, target_tokens, out_path, label):
    print(f"\n[{label}] writing up to {target_tokens:,} tokens -> {out_path}")
    arr = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=(target_tokens,))
    written = 0
    docs = 0
    last_report = 0
    for ex in dataset_iter:
        ids = enc.encode_ordinary(ex['text'])
        ids.append(EOT)
        n = len(ids)
        if written + n > target_tokens:
            n = target_tokens - written
            ids = ids[:n]
        arr[written:written + n] = ids
        written += n
        docs += 1
        if written - last_report > 2_000_000:
            pct = 100.0 * written / target_tokens
            print(f"  [{label}] {written:>12,} / {target_tokens:,} tokens  ({pct:5.1f}%)  docs: {docs:,}")
            last_report = written
        if written >= target_tokens:
            break
    arr.flush()
    print(f"[{label}] done. wrote {written:,} tokens from {docs:,} docs")
    return written

if __name__ == '__main__':
    print("Loading OpenWebText (streaming)...")
    # Try openwebtext first; fall back to c4 or wikitext if it fails
    try:
        ds = load_dataset("Skylion007/openwebtext", streaming=True, split='train', trust_remote_code=True)
    except Exception as e:
        print(f"openwebtext failed ({e}); falling back to wikitext-103")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          streaming=True, split='train')

    it = iter(ds)
    # Write train
    tokenize_stream(it, TARGET_TRAIN_TOKENS,
                    os.path.join(OUT_DIR, 'train.bin'), 'train')
    # Write val from the same iterator (next samples after train)
    tokenize_stream(it, TARGET_VAL_TOKENS,
                    os.path.join(OUT_DIR, 'val.bin'), 'val')
    print("\nDone.")

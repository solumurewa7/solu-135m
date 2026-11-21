"""
Data pipeline for Solu-135M pretraining.

Streams text and code from Hugging Face, tokenizes with the SmolLM tokenizer,
blends the two sources by token count, and writes the result to disk as a set
of fixed-size shards. Sharding keeps memory bounded: only one shard is held in
RAM while building, and at training time shards are memory-mapped so batches
are read from disk on demand rather than loading all tokens into RAM.

Run this file directly to build the shards and print a sanity check.
"""

from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
import glob
import torch
import random

# ----- Config --------------------------------------------------------------

SHARD_DIR = "shards"            # folder holding shard_000.npy, shard_001.npy, ...
SHARD_SIZE = 100_000_000        # tokens per shard (~100M)
TOTAL_TOKENS = 10_000_000_000   # total target (~10B). Lower this for a test build.
BLOCK_SIZE = 1024               # sequence length (use 256 for small local tests)
BATCH_SIZE = 32                 # sequences per batch (use 4 for small local tests)
CODE_FRACTION = 0.1             # target share of code tokens per shard
MAX_CODE_DOC_TOKENS = 2048      # drop code files longer than this
VAL_SHARDS = 1                  # number of shards held out for validation

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

# ---------------------------------------------------------------------------


def next_doc_tokens(data_iter, text_field, max_doc_tokens):
    """Pull one document from an iterator and return its token ids (+EOS).

    Returns None if the document is dropped (too long) or the stream ends,
    so the caller can simply try again.
    """
    try:
        example = next(data_iter)
    except StopIteration:
        return None
    ids = tokenizer.encode(example[text_field])
    ids.append(tokenizer.eos_token_id)
    if max_doc_tokens is not None and len(ids) > max_doc_tokens:
        return None
    return ids


def build_shards():
    """Stream, tokenize, and blend text+code into fixed-size shards on disk.

    Memory stays bounded because only one shard's tokens are held in RAM at a
    time; each shard is shuffled, saved, and freed before the next is built.
    Skips building if the shard directory already has shards.
    """
    os.makedirs(SHARD_DIR, exist_ok=True)
    if sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npy"))):
        print(f"{SHARD_DIR}/ already has shards - skipping build")
        return

    text_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                                split="train", streaming=True)
    code_dataset = load_dataset("codeparrot/codeparrot-clean",
                                split="train", streaming=True)
    text_iter = iter(text_dataset)
    code_iter = iter(code_dataset)

    total_tokens_written = 0
    shard_index = 0

    while total_tokens_written < TOTAL_TOKENS:
        shard_docs = []
        shard_tokens = 0
        code_tokens_in_shard = 0

        # Fill one shard, keeping code near CODE_FRACTION of the shard's tokens.
        while shard_tokens < SHARD_SIZE:
            want_code = code_tokens_in_shard < CODE_FRACTION * max(shard_tokens, 1)
            if want_code:
                ids = next_doc_tokens(code_iter, "content", MAX_CODE_DOC_TOKENS)
                if ids is not None:
                    code_tokens_in_shard += len(ids)
            else:
                ids = next_doc_tokens(text_iter, "text", None)

            if ids is None:
                continue
            shard_docs.append(ids)
            shard_tokens += len(ids)

        # Shuffle documents within the shard, flatten, save, then free memory.
        random.shuffle(shard_docs)
        shard_stream = []
        for doc in shard_docs:
            shard_stream.extend(doc)
        arr = np.array(shard_stream, dtype=np.uint16)
        np.save(os.path.join(SHARD_DIR, f"shard_{shard_index:03d}.npy"), arr)

        total_tokens_written += len(arr)
        code_pct = 100 * code_tokens_in_shard / len(arr)
        print(f"shard {shard_index:03d}: {len(arr):,} tokens "
              f"({code_pct:.1f}% code) | total {total_tokens_written:,}")
        shard_index += 1
        # shard_docs / shard_stream are rebound next iteration -> memory freed.


# ----- Loading (memory-mapped) ---------------------------------------------

def load_shards():
    """Memory-map every shard. Returns (train_shards, val_shards) as lists of
    numpy memmap arrays. The last VAL_SHARDS shards are held out for validation.
    Memory-mapping means tokens are read from disk on demand, so the full
    dataset never has to fit in RAM."""
    paths = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npy")))
    if not paths:
        raise FileNotFoundError(f"no shards found in {SHARD_DIR}/ - run build_shards() first")

    shards = [np.load(p, mmap_mode="r") for p in paths]
    if len(shards) > VAL_SHARDS:
        train_shards = shards[:-VAL_SHARDS]
        val_shards = shards[-VAL_SHARDS:]
    else:
        # Not enough shards to hold one out (e.g. a tiny test build): reuse all.
        train_shards = shards
        val_shards = shards
    return train_shards, val_shards


def get_batch(split="train"):
    """Sample a batch of (x, y) sequences, where y is x shifted one token ahead.

    Picks a random shard, then a random position within it. Returns two tensors
    of shape (BATCH_SIZE, BLOCK_SIZE).
    """
    shards = train_shards if split == "train" else val_shards

    x_list, y_list = [], []
    for _ in range(BATCH_SIZE):
        shard = shards[random.randrange(len(shards))]
        max_start = len(shard) - (BLOCK_SIZE + 1)   # last start that fits a full window
        start = random.randrange(max_start)
        chunk = shard[start: start + BLOCK_SIZE + 1]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])

    return torch.stack(x_list), torch.stack(y_list)


# Build (if needed) and memory-map the shards on import.
build_shards()
train_shards, val_shards = load_shards()


if __name__ == "__main__":
    n_train = sum(len(s) for s in train_shards)
    n_val = sum(len(s) for s in val_shards)
    print(f"train shards: {len(train_shards)} ({n_train:,} tokens)")
    print(f"val shards:   {len(val_shards)} ({n_val:,} tokens)")
    x, y = get_batch("train")
    print("train x shape:", x.shape)
    print("first train x row:", x[0][:10])
    print("first train y row:", y[0][:10])

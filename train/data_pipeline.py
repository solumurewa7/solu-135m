"""
Data pipeline for Solu-135M pretraining.

Streams text and code from Hugging Face, tokenizes with the SmolLM tokenizer,
blends the two sources by token count, shuffles at the document level, caches
the result to disk, and serves train/validation batches.

Running this file directly builds the token cache and prints a sanity check.
"""

from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
import torch
import random

# ----- Config --------------------------------------------------------------

TOKEN_FILE = "tokens.npy"
MAX_DOCS = 500              # text documents to pull (raise for the full run)
BLOCK_SIZE = 1024          # sequence length  (use 256 for small local tests)
BATCH_SIZE = 8             # sequences per batch (use 4 for small local tests)
CODE_FRACTION = 0.1        # target share of code tokens in the blend
MAX_CODE_DOC_TOKENS = 2048 # drop code files longer than this (keeps diversity)
VAL_FRACTION = 0.1         # tail fraction held out for validation

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

# ---------------------------------------------------------------------------


def tokenize_stream(dataset, text_field, max_docs, label, token_budget=None, max_doc_tokens=None):
    """Tokenize documents from a streaming dataset into a list of token lists.

    Each document is encoded and terminated with an EOS token. Stops when it
    hits max_docs, or (if given) when token_budget tokens have been collected.
    Documents longer than max_doc_tokens are skipped.
    """
    print(f"tokenizing {label}: ", end="")
    docs = []
    total_tokens = 0
    for example in dataset:
        ids = tokenizer.encode(example[text_field])
        ids.append(tokenizer.eos_token_id)

        if max_doc_tokens is not None and len(ids) > max_doc_tokens:
            continue

        # Check the budget before adding, so we never overshoot it.
        if token_budget is not None and total_tokens + len(ids) > token_budget:
            print(" done (hit token budget)")
            break

        docs.append(ids)
        total_tokens += len(ids)

        if len(docs) >= max_docs:
            print(" done (hit doc limit)")
            break
        if len(docs) % 50 == 0:
            print("█", end="")
    return docs


def build_token_file():
    """Build the tokenized, blended, shuffled token cache if it doesn't exist."""
    if os.path.exists(TOKEN_FILE):
        print(f"{TOKEN_FILE} already exists - skipping tokenization")
        return

    print("streaming and tokenizing...")

    text_dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    code_dataset = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True,
    )

    text_docs = tokenize_stream(text_dataset, "text", MAX_DOCS, "text")

    # Size the code budget so code ends up as CODE_FRACTION of the final blend.
    text_token_total = sum(len(d) for d in text_docs)
    code_budget = int(text_token_total * CODE_FRACTION / (1 - CODE_FRACTION))
    code_docs = tokenize_stream(
        code_dataset, "content", MAX_DOCS, "code",
        token_budget=code_budget, max_doc_tokens=MAX_CODE_DOC_TOKENS,
    )

    # Shuffle at the document level (never within a document) so the two
    # sources are interleaved while each document stays intact.
    all_docs = text_docs + code_docs
    random.seed(1337)
    random.shuffle(all_docs)

    token_stream = []
    for doc in all_docs:
        token_stream.extend(doc)

    # uint16 is enough for a 49,152-token vocab and halves the file size.
    tokens = np.array(token_stream, dtype=np.uint16)
    np.save(TOKEN_FILE, tokens)
    n_code_tokens = sum(len(d) for d in code_docs)
    print(f"saved {len(tokens)} tokens to {TOKEN_FILE} "
          f"({len(tokens) - n_code_tokens} text, {n_code_tokens} code)")


def load_tokens():
    """Load the token cache, split off a validation tail, and return
    (train_tokens, val_tokens, n_train_chunks, n_val_chunks)."""
    tokens_np = np.load(TOKEN_FILE)
    tokens = torch.from_numpy(tokens_np.astype(np.int64))   # long for embedding lookup

    # Hold out a contiguous tail so no document straddles the train/val split.
    n_val_tokens = int(len(tokens) * VAL_FRACTION)
    split_point = len(tokens) - n_val_tokens
    train_tokens = tokens[:split_point]
    val_tokens = tokens[split_point:]

    n_train_chunks = len(train_tokens) // (BLOCK_SIZE + 1)
    n_val_chunks = len(val_tokens) // (BLOCK_SIZE + 1)
    return train_tokens, val_tokens, n_train_chunks, n_val_chunks


def get_batch(split="train"):
    """Sample a batch of (x, y) sequences, where y is x shifted one token ahead.

    Returns two tensors of shape (BATCH_SIZE, BLOCK_SIZE).
    """
    if split == "train":
        data, n = train_tokens, n_train_chunks
    else:
        data, n = val_tokens, n_val_chunks

    chunk_indices = torch.randint(0, n, (BATCH_SIZE,))

    x_list, y_list = [], []
    for idx in chunk_indices:
        start = idx * (BLOCK_SIZE + 1)
        chunk = data[start: start + BLOCK_SIZE + 1]
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])

    return torch.stack(x_list), torch.stack(y_list)


# Build the cache and load it on import, so training can just call get_batch.
build_token_file()
train_tokens, val_tokens, n_train_chunks, n_val_chunks = load_tokens()


if __name__ == "__main__":
    x, y = get_batch("train")
    xv, _ = get_batch("val")
    print("train x shape:", x.shape)
    print("val x shape:", xv.shape)
    print("n_train_chunks:", n_train_chunks)
    print("n_val_chunks:", n_val_chunks)
    print("first train x row:", x[0][:10])
    print("first train y row:", y[0][:10])

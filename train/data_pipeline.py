from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
import torch
import random

# -----------------------------------------------------

TOKEN_FILE = "tokens.npy"
MAX_DOCS = 500
BLOCK_SIZE = 1024 # test = 256 | main = 1024
BATCH_SIZE = 8 # test = 4 | main = 8
CODE_FRACTION = 0.1
MAX_CODE_DOC_TOKENS = 2048
VAL_FRACTION = 0.1

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

# -----------------------------------------------------

def tokenize_stream(dataset, text_field, max_docs, label, token_budget=None, max_doc_tokens=None):
    print(f"tokenizing {label}: ", end="")
    docs = []
    total_tokens = 0
    for i, example in enumerate(dataset):
        ids = tokenizer.encode(example[text_field])
        ids.append(tokenizer.eos_token_id)

        if max_doc_tokens is not None and len(ids) > max_doc_tokens:
            continue

        if token_budget is not None and total_tokens + len(ids) > token_budget:
            print(" done (hit token budget)")
            break

        docs.append(ids)
        total_tokens += len(ids)

        if len(docs) >= max_docs:
            print(" done (hit doc limit)")
            break
        if len(docs) != 0 and len(docs) % 50 == 0:
            print("█", end="")
    return docs

# -----------------------------------------------------

def build_token_file(): # Builds token file if it doesnt exist. Blends text + code, saves to disk
    if os.path.exists(TOKEN_FILE):
        print(f"{TOKEN_FILE} already exists - skipping tokenization")
        return

    print("streaming and tokenizing...")

    n_text_docs = MAX_DOCS

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

    text_docs = tokenize_stream(text_dataset, "text", n_text_docs, "text")
    text_token_total = sum(len(d) for d in text_docs)
    code_budget = int(text_token_total * CODE_FRACTION / (1 - CODE_FRACTION))

    code_docs = tokenize_stream(code_dataset, "content", MAX_DOCS, "code", token_budget=code_budget, max_doc_tokens=MAX_CODE_DOC_TOKENS)

    all_docs = text_docs + code_docs
    random.seed(1337)
    random.shuffle(all_docs)

    token_stream = []
    for doc in all_docs:
        token_stream.extend(doc)

    tokens = np.array(token_stream, dtype=np.uint16)
    np.save(TOKEN_FILE, tokens)
    n_code_tokens = sum(len(d) for d in code_docs)
    print(f"saved {len(tokens)} tokens to {TOKEN_FILE} ({len(tokens) - n_code_tokens} text, {n_code_tokens} code)")

# -----------------------------------------------------

def load_tokens():  # loads the tokens from the token file and converts to long. returns number of chuncks and all tokens
    tokens_np = np.load(TOKEN_FILE)
    tokens = torch.from_numpy(tokens_np.astype(np.int64))

    n_val_tokens = int(len(tokens) * VAL_FRACTION)
    split_point = len(tokens) - n_val_tokens

    train_tokens = tokens[:split_point]
    val_tokens = tokens[split_point:]

    n_train_chunks = len(train_tokens) // (BLOCK_SIZE + 1)
    n_val_chunks = len(val_tokens) // (BLOCK_SIZE + 1)

    return train_tokens, val_tokens, n_train_chunks, n_val_chunks
# -----------------------------------------------------

def get_batch(split="train"):  # gets batch of tokens for training and returns x, y, array of all the tokens and answers.
    if split == "train":
        data = train_tokens
        n = n_train_chunks
    else:
        data = val_tokens
        n = n_val_chunks

    chunk_indices = torch.randint(0, n, (BATCH_SIZE,))

    x_list = []
    y_list = []
    for idx in chunk_indices:
        start = idx * (BLOCK_SIZE + 1)
        chunk = data[start : start + BLOCK_SIZE + 1]
        x_list.append(chunk[:-1])
        y_list.append(chunk[1:])

    x = torch.stack(x_list)
    y = torch.stack(y_list)
    return x, y

# -----------------------------------------------------

build_token_file()
train_tokens, val_tokens, n_train_chunks, n_val_chunks = load_tokens()

if __name__ == "__main__":
    x, y = get_batch("train")
    xv, yv = get_batch("val")
    print("train x shape:", x.shape)
    print("val x shape:", xv.shape)
    print("n_train_chunks:", n_train_chunks)
    print("n_val_chunks:", n_val_chunks)
    print("first train x row:", x[0][:10])
    print("first train y row:", y[0][:10])

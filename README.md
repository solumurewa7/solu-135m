# Solu-135M

A 135M-parameter decoder-only transformer language model, built from scratch in PyTorch.

Solu-135M implements a modern LLM architecture end to end: grouped-query attention, rotary position embeddings, RMSNorm, SwiGLU, and tied embeddings. The model, the data pipeline, and the training loop are all hand-written. PyTorch is used for tensor operations and autograd only, not for any high-level model, attention, or transformer-block abstractions. The goal is to understand how every piece of a current-generation language model actually works by building it rather than importing it.

> **Status: work in progress.** The architecture and training pipeline are complete and verified. The full pretraining run has not been done yet, so there are no trained weights or benchmark numbers in this repo yet. The [roadmap](#roadmap) below tracks what's done and what's next.

---

## What's here

| File | What it does |
|------|--------------|
| `model.py` | The full model: `Solu`, plus `Block`, `RMSNorm`, `SwiGLU`, and `GroupedQueryAttention`. Shared by training and (future) inference. |
| `train/data_pipeline.py` | Streams and tokenizes the training data, blends text + code by token ratio, caches to disk, and serves train/val batches. |
| `train/train.py` | The training loop: LR schedule, gradient clipping, mixed precision, checkpointing, and logging. |
| `inference/` | Reserved for the Pi-portable inference code (empty for now). |

---

## Architecture

Parameter count: **134,515,008**.

| Component | Value |
|-----------|-------|
| Layers | 30 |
| Model dim | 576 |
| FFN hidden dim | 1536 |
| Query heads | 9 |
| KV heads | 3 (grouped-query attention) |
| Head dim | 64 |
| Vocab size | 49,152 |
| Context length | 1024 |
| Position encoding | RoPE (θ=10000, split-half pairing) |
| Normalization | RMSNorm (pre-norm, eps=1e-5) |
| FFN | SwiGLU |
| Embeddings | Tied (input embedding reused as output projection) |

**Design notes**
- **Grouped-query attention (GQA):** 9 query heads share 3 KV heads, cutting the key/value projection size while keeping query expressiveness. The repository keeps both a hand-written attention path and a fast path through `F.scaled_dot_product_attention`, verified to produce identical output; the hand-written version exists to make the mechanism legible, the fast path is used for training.
- **RoPE in float32:** the rotary computation is forced to float32 even under bfloat16 autocast, because bfloat16 loses positional precision at longer context lengths. This was a deliberate fix, not the default.
- **GPT-2-style initialization:** all linear and embedding weights init at std 0.02; the residual-writing projections are additionally scaled by `0.02 / sqrt(2 * n_layers)` to keep activation variance stable across 30 residual additions, which prevents the loss blow-up that an unscaled init produces at this depth.
- **Tied embeddings:** the input embedding table is reused as the output projection, saving ~28M parameters.

---

## Data

Pretraining data is a blend, streamed from Hugging Face and tokenized with the SmolLM BPE tokenizer (49,152-token vocabulary):

- **~90% text:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) — educational web text, for language and general knowledge.
- **~10% code:** [CodeParrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) — Python source, for code familiarity.

The pipeline makes several engineering choices that matter at scale:
- **Blend controlled by token count, not document count,** so the ratio stays accurate regardless of how long individual documents are.
- **Length filtering** on code files, so a handful of huge auto-generated files can't dominate the code portion.
- **Document-level shuffling** (never within a document, which would destroy the next-token prediction signal).
- **Contiguous validation split** (held-out tail, so no document leaks between train and val).
- **On-disk token cache,** so tokenization runs once and the expensive step never repeats across training runs.

---

## Training

The training loop (`train/train.py`) includes:

- **LR schedule:** linear warmup then cosine decay to a 10% floor.
- **Gradient clipping** at max-norm 1.0.
- **Mixed precision:** bfloat16 autocast (no gradient scaler needed with bf16).
- **Flash attention** and **`torch.compile`** for throughput, both with safe fallback where unsupported.
- **Checkpointing:** a rolling checkpoint for crash recovery plus a separate best-by-validation-loss checkpoint, both written in a way that survives `torch.compile`'s weight renaming.
- **Logging:** TensorBoard scalars for train loss, val loss, learning rate, and gradient norm.

Planned run: ~10B tokens on a single rented cloud GPU, benchmarked against GPT-2 124M.

---

## Running it

```bash
pip install -r requirements.txt

# Build the token cache and sanity-check the data pipeline
python train/data_pipeline.py

# Train
python train/train.py
```

The config in `train/data_pipeline.py` is set for the full run (block size 1024, batch size 8). For quick local testing on a small GPU, drop them to 256 / 4.

---

## Roadmap

- [x] Model architecture (hand-written, verified forward pass and parameter count)
- [x] Data pipeline (streaming, blend, shuffle, train/val split, caching)
- [x] Training loop (LR schedule, clipping, mixed precision, checkpointing, logging)
- [ ] Full pretraining run on cloud GPU
- [ ] Evaluation vs GPT-2 124M + loss curves posted here
- [ ] Trained weights published to the Hugging Face Hub
- [ ] Fine-tuning for downstream use

---

## License

[MIT](LICENSE) © 2026 Oluwaseyi (Seyi) Olumurewa
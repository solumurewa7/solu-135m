# Solu-135M

A 135M-parameter decoder-only transformer language model, built from scratch in PyTorch.

Solu-135M mirrors the architecture of [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M): grouped-query attention, rotary position embeddings, RMSNorm, SwiGLU, and tied embeddings. Everything here (the model, the data pipeline, the training loop) is hand-written to understand how a modern LLM works end to end, using PyTorch only for tensor ops and autograd rather than any high-level model library.

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

Verified parameter count: **134,515,008** (matches SmolLM-135M exactly).

| Component | Value |
|-----------|-------|
| Layers | 30 |
| Model dim | 576 |
| FFN hidden dim | 1536 |
| Query heads | 9 |
| KV heads | 3 (grouped-query attention) |
| Head dim | 64 |
| Vocab size | 49,152 (SmolLM tokenizer) |
| Context length | 1024 |
| Position encoding | RoPE (θ=10000, split-half pairing) |
| Normalization | RMSNorm (pre-norm, eps=1e-5) |
| FFN | SwiGLU |
| Embeddings | Tied (input embedding reused as output projection) |

**Design notes**
- **Grouped-query attention (GQA):** 9 query heads share 3 KV heads, cutting the KV projection size while keeping query expressiveness.
- **RoPE in float32:** the rotary computation is forced to float32 even under bfloat16 autocast, because bfloat16 loses positional precision at longer context lengths.
- **Flash attention, switchable:** attention runs through `F.scaled_dot_product_attention` by default, but a hand-written attention path is kept alongside it (`use_flash=False`) and verified to produce identical output. The hand-built version is there to make the mechanism legible, not for speed.
- **GPT-2-style init:** all linear/embedding weights init at std 0.02; residual-writing projections are scaled by `0.02 / sqrt(2 * n_layers)` to keep activations stable through 30 residual additions.

---

## Data

Pretraining data is a blend, streamed from Hugging Face and tokenized with the SmolLM tokenizer:

- **~90% text:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) — educational web text, for language and general knowledge.
- **~10% code:** [CodeParrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) — Python source, for code familiarity.

The blend is controlled **by token count**, not document count, so the ratio is accurate regardless of how long individual documents are. The pipeline also filters out pathologically long code files, shuffles at the document level (never within a document, which would destroy the next-token signal), holds out a contiguous 10% validation split with no leakage, and caches the tokenized array to disk so tokenization happens once.

---

## Training

The training loop (`train/train.py`) includes:

- **LR schedule:** linear warmup then cosine decay to a 10% floor.
- **Gradient clipping** at max-norm 1.0.
- **Mixed precision:** bfloat16 autocast (no gradient scaler needed with bf16).
- **Flash attention** and **`torch.compile`** for throughput (compile falls back safely where unsupported).
- **Checkpointing:** a rolling checkpoint for crash recovery plus a separate best-by-validation-loss checkpoint, both compile-safe.
- **Logging:** TensorBoard scalars for train loss, val loss, learning rate, and gradient norm.

Planned run: ~10B tokens on a single rented cloud GPU, with GPT-2 124M as the benchmark comparison point.

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

- [x] Model architecture (verified against SmolLM-135M, exact param count)
- [x] Data pipeline (streaming, blend, shuffle, train/val split, caching)
- [x] Training loop (LR schedule, clipping, mixed precision, checkpointing, logging)
- [ ] Full pretraining run on cloud GPU
- [ ] Evaluation vs GPT-2 124M + loss curves posted here
- [ ] Trained weights published to the Hugging Face Hub
- [ ] Fine-tuning for downstream use

---

## License

[MIT](LICENSE) © 2026 Oluwaseyi (Seyi) Olumurewa

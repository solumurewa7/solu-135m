# Solu-135M

A 135M-parameter decoder-only transformer language model, built from scratch in PyTorch, designed to run **fully on-device on a Raspberry Pi 4**.

Solu-135M is the language brain for a personal desk assistant. The goal is a small, efficient model that runs locally on constrained hardware, understands a spoken request, and turns it into a structured action: answer directly, trigger a tool, or hand a harder task off to a larger model. Everything here (the model, the data pipeline, the training loop) is hand-written. PyTorch provides tensor ops and autograd only, with no high-level model, attention, or transformer-block libraries. The point is to understand and control every part of the model, and to keep it small enough to deploy at the edge rather than in the cloud.

> **Status: work in progress.** The architecture and training pipeline are complete and verified. The base pretraining run has not been done yet, so there are no trained weights or benchmark numbers in this repo yet. The [roadmap](#roadmap) below tracks what's done and what's next.

---

## Why a 135M model

Most capable assistants run in the cloud on large models. This project goes the other way: build a model small enough to run entirely on a Raspberry Pi 4, with no network dependency for the common case. That constraint drives every design choice here, grouped-query attention to shrink the KV cache, tied embeddings to cut parameters, and an architecture chosen for inference efficiency on small hardware.

A model this size will not match a large cloud assistant on open-ended reasoning, and it is not meant to. Its job is to handle the frequent, latency-sensitive requests locally and instantly, and to recognize when a task is beyond it and escalate to a larger model. Fast and local for the common case, with a path to more capability when a task actually needs it.

---

## What's here

| File | What it does |
|------|--------------|
| `model.py` | The full model: `Solu`, plus `Block`, `RMSNorm`, `SwiGLU`, and `GroupedQueryAttention`. Shared by training and inference. |
| `train/data_pipeline.py` | Streams and tokenizes training data, blends text + code by token ratio, caches to disk, and serves train/val batches. |
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
- **Grouped-query attention (GQA):** 9 query heads share 3 KV heads, cutting the key/value projection and KV-cache size while keeping query resolution. Smaller KV cache matters directly for running on a Pi. The repo keeps both a hand-written attention path and a fused `F.scaled_dot_product_attention` path, verified to produce identical output; the hand-written version makes the mechanism legible, the fused path is used for training.
- **RoPE in float32:** the rotary computation is forced to float32 even under bfloat16 autocast, because bfloat16 loses positional precision at longer context lengths. A deliberate fix, not the default.
- **GPT-2-style initialization:** all linear and embedding weights init at std 0.02; the residual-writing projections are additionally scaled by `0.02 / sqrt(2 * n_layers)` to keep activation variance stable across 30 residual additions, which prevents the loss blow-up an unscaled init produces at this depth.
- **Tied embeddings:** the input embedding table is reused as the output projection, saving ~28M parameters.

---

## Data

Base pretraining data is a blend, streamed from Hugging Face and tokenized with the SmolLM BPE tokenizer (49,152-token vocabulary):

- **~90% text:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) — educational web text, for language and general knowledge.
- **~10% code:** [CodeParrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) — Python source, for code familiarity.

The pipeline makes several choices that matter at scale: the blend is controlled **by token count, not document count**, so the ratio stays accurate regardless of document length; long code files are length-filtered so a few huge files can't dominate; shuffling is **document-level** (never within a document, which would destroy the next-token signal); the validation split is a **contiguous tail** so no document leaks between train and val; and the tokenized array is **cached to disk** so tokenization runs once.

Base pretraining gives the model language and code familiarity. The assistant behavior (structured intent / action / tool-call / escalate outputs) is taught in a later fine-tuning stage, where the structured output format and its reserved tokens are introduced. Keeping that out of base pretraining means the format can be designed once the assistant's task set is finalized, without re-running the base.

---

## Training

The training loop (`train/train.py`) includes:

- **LR schedule:** linear warmup then cosine decay to a 10% floor.
- **Gradient clipping** at max-norm 1.0.
- **Mixed precision:** bfloat16 autocast (no gradient scaler needed with bf16).
- **Flash attention** and **`torch.compile`** for throughput, both with safe fallback where unsupported.
- **TF32** matmuls enabled on supported GPUs.
- **Checkpointing:** a rolling checkpoint for crash recovery plus a separate best-by-validation-loss checkpoint, both written to survive `torch.compile`'s weight renaming.
- **Logging:** TensorBoard scalars for train loss, val loss, learning rate, and gradient norm.

Planned base run: ~10B tokens on a single rented cloud GPU, benchmarked against GPT-2 124M.

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
- [ ] Base pretraining run on cloud GPU
- [ ] Evaluation vs GPT-2 124M + loss curves posted here
- [ ] Trained weights published to the Hugging Face Hub
- [ ] Fine-tune for assistant use (structured intent / action / tool-call / escalate outputs)
- [ ] Export + quantize for on-device inference on a Raspberry Pi 4; measure tokens/sec

---

## License

[MIT](LICENSE) © 2026 Oluwaseyi (Seyi) Olumurewa

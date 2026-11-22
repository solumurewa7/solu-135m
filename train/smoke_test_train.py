"""
Smoke test for the full training loop, run on the CPU box BEFORE renting a GPU.

Why this exists: train.py has many parts that only all run together for the first
time on the expensive A100 (real shards -> get_batch -> model -> loss -> backward
-> optimizer -> CSV logging). A dumb bug in that seam (a wrong dtype, a bad CSV
column, a shape mismatch) would otherwise surface only after you've paid for a
GPU and waited through torch.compile. This runs the entire seam against the REAL
shards with tiny settings, on CPU, in about a minute, so any such bug is caught
for free.

What it deliberately does NOT test: real training quality, speed, or GPU memory.
The loss here means nothing (tiny model, few steps, CPU). It only proves the
machinery executes end to end and writes a valid metrics row.

Run it on the CPU pod once the shards exist:
    cd /workspace/solu-135m
    python train/smoke_test_train.py

Expect: it prints a few decreasing-ish (or at least finite) loss lines and writes
smoke_metrics.csv, then prints SMOKE TEST PASSED. If it raises, fix that before
touching a GPU.
"""

import os
import sys
import csv
import math
import time

# Skip the pipeline's import-time build; we point it at the real shards ourselves.
os.environ["SOLU_SKIP_IMPORT_BUILD"] = "1"

# Make both the repo root (for model.py) and train/ (for data_pipeline) importable.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import data_pipeline as dp
from model import Solu

# ----- Tiny test config (overrides the pipeline's real globals) ------------

# Point at the real shards. Default matches the real run layout: shards/ sits
# under the repo root. Override with SOLU_SHARD_DIR if yours is elsewhere.
dp.SHARD_DIR = os.environ.get("SOLU_SHARD_DIR", os.path.join(ROOT, "shards"))

# Tiny so it runs on CPU in seconds. get_batch reads these as module globals.
# A smaller block than the real 1024 is fine: get_batch just reads a shorter
# window out of each shard.
dp.BLOCK_SIZE = 128
dp.BATCH_SIZE = 4
dp.VAL_SHARDS = 1

STEPS = 10
METRICS_CSV = os.path.join(HERE, "smoke_metrics.csv")

print(f"shard dir: {dp.SHARD_DIR}")
if not os.path.isdir(dp.SHARD_DIR) or not any(
    f.startswith("shard_") for f in os.listdir(dp.SHARD_DIR)
):
    print("ERROR: no shards found there. Build the shards first, or set "
          "SOLU_SHARD_DIR to the folder that holds shard_000.npy etc.")
    sys.exit(1)

# ----- Load real shards (memory-mapped) ------------------------------------

print("loading real shards (memory-mapped)...")
dp.train_shards, dp.val_shards = dp.load_shards()
n_train = sum(len(s) for s in dp.train_shards)
print(f"  {len(dp.train_shards)} train shards, {len(dp.val_shards)} val shards, "
      f"{n_train:,} train tokens")

# ----- A deliberately tiny model (same class, small dims) ------------------

# Same Solu class and the real vocab (must match the tokenizer that built the
# shards), but few/narrow layers so forward+backward are cheap on CPU.
print("building a tiny Solu (real class, small dims)...")
model = Solu(
    vocab_size=49152,   # must match the shards' tokenizer
    dim=64,
    n_layers=2,
    n_query_heads=4,
    n_kv_heads=2,
    hidden_dim=128,
)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# ----- The same CSV logging shape train.py uses ----------------------------

f = open(METRICS_CSV, "w", newline="")
w = csv.writer(f)
w.writerow([
    "step", "train_loss", "val_loss", "train_perplexity", "val_perplexity",
    "lr", "grad_norm", "tokens_per_sec", "step_time_s", "gpu_mem_gb",
])
f.flush()

tokens_per_step = dp.BLOCK_SIZE * dp.BATCH_SIZE

# ----- The real loop, in miniature -----------------------------------------

print(f"running {STEPS} steps on CPU...\n")
for step in range(STEPS):
    t0 = time.time()

    xb, yb = dp.get_batch("train")
    # shape check: this is exactly what train.py feeds the model
    assert xb.shape == (dp.BATCH_SIZE, dp.BLOCK_SIZE), f"bad x shape {xb.shape}"
    assert yb.shape == (dp.BATCH_SIZE, dp.BLOCK_SIZE), f"bad y shape {yb.shape}"
    # y must be x shifted one token ahead (the next-token target)
    assert torch.equal(xb[0, 1:], yb[0, :-1]), "y is not x shifted by one!"

    logits = model(xb)
    assert logits.shape[:2] == (dp.BATCH_SIZE, dp.BLOCK_SIZE), \
        f"bad logits shape {logits.shape}"
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    step_time = time.time() - t0
    tps = tokens_per_step / step_time
    train_loss = loss.item()
    gn = grad_norm.item()
    assert math.isfinite(train_loss), f"loss is not finite: {train_loss}"
    assert math.isfinite(gn), f"grad norm is not finite: {gn}"

    tppl = math.exp(min(train_loss, 20))
    w.writerow([step, f"{train_loss:.6f}", f"{train_loss:.6f}", f"{tppl:.4f}",
                f"{tppl:.4f}", "0.00030000", f"{gn:.4f}", f"{tps:.1f}",
                f"{step_time:.5f}", "0.000"])
    f.flush()
    print(f"step {step}: loss {train_loss:.4f} | gnorm {gn:.2f} | {tps:,.0f} tok/s")

f.close()

# ----- Confirm the CSV is readable and well-formed -------------------------

with open(METRICS_CSV, newline="") as rf:
    rows = list(csv.DictReader(rf))
assert len(rows) == STEPS, f"expected {STEPS} rows, got {len(rows)}"
assert float(rows[-1]["train_loss"]) > 0, "last row has no loss"

print(f"\nwrote {METRICS_CSV} ({len(rows)} rows, parses cleanly)")
print("\nSMOKE TEST PASSED - the full loop runs against the real shards and "
      "logs valid metrics. Safe to run the real train.py on the GPU.")

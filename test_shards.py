"""
Quick local sanity test for the sharded data pipeline.

Builds a tiny set of shards into a temporary folder and confirms that building,
memory-mapped loading, and batching all work and that the x/y shift is correct.
Does NOT touch your real config or the real shards/.

Run from the repo root:
    python test_shards.py
"""

import os
# Must be set BEFORE importing the pipeline, so its import-time build is skipped
# and this test controls the (tiny) config and build instead.
os.environ["SOLU_SKIP_IMPORT_BUILD"] = "1"

import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train"))

import data_pipeline as dp

# --- Tiny test values ---
dp.SHARD_DIR = "shards_test"
dp.SHARD_SIZE = 500_000         # ~0.5M tokens per shard
dp.TOTAL_TOKENS = 1_500_000     # ~1.5M total -> about 3 small shards
dp.BLOCK_SIZE = 256
dp.BATCH_SIZE = 4
dp.VAL_SHARDS = 1

if os.path.exists(dp.SHARD_DIR):
    shutil.rmtree(dp.SHARD_DIR)

print("building tiny test shards (streams a little data from HF)...")
dp.build_shards()

print("\nloading shards (memory-mapped)...")
dp.train_shards, dp.val_shards = dp.load_shards()

n_train = sum(len(s) for s in dp.train_shards)
n_val = sum(len(s) for s in dp.val_shards)
print(f"train shards: {len(dp.train_shards)} ({n_train:,} tokens)")
print(f"val shards:   {len(dp.val_shards)} ({n_val:,} tokens)")

print("\nsampling a batch...")
x, y = dp.get_batch("train")
print("train x shape:", x.shape, "(expect [4, 256])")
print("val x shape:  ", dp.get_batch("val")[0].shape)

print("\nshift check (y should be x shifted one token ahead):")
print("x[0][:10]:", x[0][:10].tolist())
print("y[0][:10]:", y[0][:10].tolist())
print("shift correct:", x[0][1:10].tolist() == y[0][:9].tolist())

print("\ncleaning up test shards...")
shutil.rmtree(dp.SHARD_DIR)
print("done. (your real shards/ and config were not touched)")

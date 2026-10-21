# -----------------------------------------------------
# Solu-135M training script
# Run from the repo root:  python train/train.py
# -----------------------------------------------------

import sys
import os
# Add repo root to the import path so "from model import Solu" works
# even though this file lives in train/ and model.py lives at the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from model import Solu
from data_pipeline import get_batch
import math
from torch.utils.tensorboard import SummaryWriter
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import logging
torch._logging.set_logs(dynamo=logging.ERROR)

# -----------------------------------------------------

torch.set_float32_matmul_precision('high')
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device: ", device)
model = Solu(
    vocab_size=49152,
    dim=576,
    n_layers=30,
    n_query_heads=9,
    n_kv_heads=3,
    hidden_dim=1536,
)
model = model.to(device)

CHECKPOINT_FILE = "checkpoint.pt"
BEST_CHECKPOINT_FILE = "best_checkpoint.pt"
best_val_loss = float('inf')
TOTAL_STEPS = 50
SAVE_EVERY = 25
MAX_LR = 3e-4
WARMUP_STEPS = 10
LR_DECAY_STEPS = TOTAL_STEPS
MAX_GRAD_NORM = 1.0
log_steps = []
log_train_loss = []
log_val_loss = []
USE_COMPILE = True

# -----------------------------------------------------

with torch.no_grad():
    xb, yb = get_batch()
    xb = xb.to(device)
    logits = model(xb)
    print("logit stats -> mean:", logits.mean().item(), "std:", logits.std().item(), "max abs:", logits.abs().max().item())

# -----------------------------------------------------

def unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model

# -----------------------------------------------------

def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > LR_DECAY_STEPS:
        return MAX_LR * 0.1

    progress = (step - WARMUP_STEPS) / (LR_DECAY_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MAX_LR * 0.1 + coeff * (MAX_LR - MAX_LR * 0.1)

# -----------------------------------------------------

print("model built, parameters: ", sum(p.numel() for p in model.parameters()) / 1e6, "M")
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)
model.train()
writer = SummaryWriter()
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
print("bfloat16 supported:", use_bf16)

# -----------------------------------------------------

@torch.no_grad()
def estimate_val_loss(n_batches=10):
    model.eval()
    losses = []
    for _ in range(n_batches):
        xb, yb = get_batch("val")
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits = model(xb)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            yb = yb.view(B * T)
            loss = F.cross_entropy(logits, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# -----------------------------------------------------

def save_checkpoint(model, optimizer, step, best_val_loss, filename=CHECKPOINT_FILE):
    checkpoint = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    try:
        torch.save(checkpoint, filename)
        print(f"  saved checkpoint at step {step} -> {filename}")
    except Exception as e:
        print(f"  WARNING: failed to save checkpoint to {filename}: {e}")

# -----------------------------------------------------

def load_checkpoint(model, optimizer):
    if not os.path.exists(CHECKPOINT_FILE):
        print("no checkpoint found, starting fresh")
        return 0, float('inf')

    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=True)
    unwrap(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    print(f"resumed from checkpoint at step {step}, continuing from {step + 1}")
    return step + 1, best_val_loss

# -----------------------------------------------------

start_step, best_val_loss = load_checkpoint(model, optimizer)

if USE_COMPILE:
    try:
        model = torch.compile(model)
        print("model compiled")
    except Exception as e:
        print(f"torch.compile failed, running uncompiled: {e}")

for step in range(start_step, TOTAL_STEPS):

    # -------------------------

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # -------------------------

    xb, yb = get_batch()
    xb, yb = xb.to(device), yb.to(device)

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        logits = model(xb)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        yb = yb.view(B * T)
        loss = F.cross_entropy(logits, yb)

    # -------------------------

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    # -------------------------

    if step % 5 == 0:
        val_loss = estimate_val_loss()
        print(f"step {step}: train loss {loss.item():.4f} | val loss {val_loss:.4f} | lr {lr:.6f} | grad_norm {grad_norm:.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, step, best_val_loss, BEST_CHECKPOINT_FILE)
        log_steps.append(step)
        log_train_loss.append(loss.item())
        log_val_loss.append(val_loss)
        writer.add_scalar("loss/train", loss.item(), step)
        writer.add_scalar("loss/val", val_loss, step)
        writer.add_scalar("lr", lr, step)
        writer.add_scalar("grad_norm", grad_norm, step)

    # -------------------------

    if step % SAVE_EVERY == 0 and step > 0:
        save_checkpoint(model, optimizer, step, best_val_loss)

writer.close()

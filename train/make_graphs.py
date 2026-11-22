"""
Render training graphs for the Solu-135M findings deck.

Reads the two CSVs the training run produced (metrics.csv from train.py and,
optionally, gpu_metrics.csv from gpu_logger.py) and writes a folder of PNGs plus
a one-page summary of headline numbers. This runs on your laptop AFTER training,
not on the GPU. Rendering plots during a run would stall the GPU and waste money,
so the rule is: log cheap numbers to CSV on the box, draw the pictures later here.

Usage:
    python train/make_graphs.py                      # expects metrics.csv in cwd
    python train/make_graphs.py --metrics path/to/metrics.csv --gpu path/to/gpu_metrics.csv --out graphs

Each chart below is one answer to a question someone at NVIDIA would actually ask
about a training run. The comments say what the question is and how to read it.
"""

import argparse
import csv
import os
import math

import matplotlib
matplotlib.use("Agg")   # render to files, no display needed
import matplotlib.pyplot as plt


# ----- A consistent look for every chart ----------------------------------

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.frameon": False,
})

BLUE = "#1f6feb"
GREEN = "#2da44e"
ORANGE = "#e3681c"
PURPLE = "#8250df"
RED = "#cf222e"
GRAY = "#57606a"


# ----- CSV loading ---------------------------------------------------------

def load_csv(path):
    """Read a CSV into a dict of column_name -> list of floats (strings kept as
    strings where they don't parse). Returns {} if the file is missing/empty."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except (ValueError, TypeError):
                cols[k].append(v)
    return cols


# ----- Small helpers -------------------------------------------------------

def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _smooth(values, window):
    """Simple moving average, used to show the trend under a noisy curve."""
    if window <= 1 or len(values) < window:
        return None
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo: i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


# ----- The charts ----------------------------------------------------------

def chart_loss(m, out_dir):
    """Q: Is the model learning, and is it overfitting?
    Train and val loss on the same axes. If val pulls away above train, that's
    overfitting. A smoothed train line sits on top so the trend is readable
    under the step-to-step noise."""
    steps = m["step"]
    fig, ax = plt.subplots()
    ax.plot(steps, m["train_loss"], color=BLUE, alpha=0.35, linewidth=1, label="train (raw)")
    sm = _smooth(m["train_loss"], 20)
    if sm:
        ax.plot(steps, sm, color=BLUE, linewidth=2, label="train (smoothed)")
    ax.plot(steps, m["val_loss"], color=ORANGE, linewidth=2, label="val")
    ax.set_title("Training & validation loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.legend()
    _save(fig, out_dir, "01_loss.png")


def chart_perplexity(m, out_dir):
    """Q: How good is the model in human-readable terms?
    Perplexity is exp(loss): roughly how many tokens the model is 'choosing
    between' at each step. Lower is better. Easier to talk about than raw loss."""
    steps = m["step"]
    fig, ax = plt.subplots()
    ax.plot(steps, m["train_perplexity"], color=BLUE, linewidth=2, label="train")
    ax.plot(steps, m["val_perplexity"], color=ORANGE, linewidth=2, label="val")
    ax.set_title("Perplexity (exp of loss)")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.legend()
    _save(fig, out_dir, "02_perplexity.png")


def chart_lr(m, out_dir):
    """Q: Did the learning-rate schedule do what we designed?
    Should show the linear warmup ramp, then the smooth cosine decay down to the
    floor. If this looks wrong, the schedule is misconfigured."""
    fig, ax = plt.subplots()
    ax.plot(m["step"], m["lr"], color=PURPLE, linewidth=2)
    ax.set_title("Learning-rate schedule (warmup + cosine decay)")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    _save(fig, out_dir, "03_learning_rate.png")


def chart_grad_norm(m, out_dir):
    """Q: Is training numerically stable?
    Gradient norm before clipping. Occasional spikes are normal; a sustained
    climb or frequent hits at the clip ceiling (1.0 here) means instability.
    The clip line is drawn so you can see how often we're hitting it."""
    fig, ax = plt.subplots()
    ax.plot(m["step"], m["grad_norm"], color=GREEN, linewidth=1.2, alpha=0.8)
    ax.axhline(1.0, color=RED, linestyle="--", linewidth=1, label="clip threshold (1.0)")
    ax.set_title("Gradient norm (pre-clip)")
    ax.set_xlabel("step")
    ax.set_ylabel("global grad norm")
    ax.legend()
    _save(fig, out_dir, "04_grad_norm.png")


def chart_throughput(m, out_dir):
    """Q: How fast are we training, and is the speed steady?
    Tokens/sec per step. The flat median line is the honest headline number for
    the deck. Early steps are slower (compile warmup), so the median ignores
    that better than the mean."""
    steps = m["step"]
    tps = m["tokens_per_sec"]
    fig, ax = plt.subplots()
    ax.plot(steps, tps, color=BLUE, linewidth=1, alpha=0.5, label="per step")
    sm = _smooth(tps, 20)
    if sm:
        ax.plot(steps, sm, color=BLUE, linewidth=2, label="smoothed")
    med = sorted(tps)[len(tps) // 2]
    ax.axhline(med, color=GRAY, linestyle="--", linewidth=1,
               label=f"median {med:,.0f} tok/s")
    ax.set_title("Throughput")
    ax.set_xlabel("step")
    ax.set_ylabel("tokens / second")
    ax.legend()
    _save(fig, out_dir, "05_throughput.png")


def chart_step_time(m, out_dir):
    """Q: How long does one step take, and are there stalls?
    Step time in milliseconds. Spikes are usually checkpoint saves or data
    hiccups. A slow upward drift would mean something is leaking."""
    steps = m["step"]
    ms = [s * 1000 for s in m["step_time_s"]]
    fig, ax = plt.subplots()
    ax.plot(steps, ms, color=ORANGE, linewidth=1, alpha=0.7)
    ax.set_title("Step time")
    ax.set_xlabel("step")
    ax.set_ylabel("milliseconds per step")
    _save(fig, out_dir, "06_step_time.png")


def chart_gpu_mem_from_train(m, out_dir):
    """Q: How much VRAM did the model use? (as seen by PyTorch)
    This is torch's own max-allocated figure, logged by train.py. It's the number
    that tells you whether the batch size fits and how much headroom is left."""
    fig, ax = plt.subplots()
    ax.plot(m["step"], m["gpu_mem_gb"], color=PURPLE, linewidth=2)
    ax.set_title("GPU memory used by the model (PyTorch max-allocated)")
    ax.set_xlabel("step")
    ax.set_ylabel("GB")
    _save(fig, out_dir, "07_gpu_mem_pytorch.png")


def chart_gpu_util(g, out_dir):
    """Q: Are we actually using the GPU we're paying for?
    GPU utilization % over wall-clock time, from nvidia-smi. Sustained high
    utilization (ideally 90%+) means the GPU is compute-bound and we're getting
    our money's worth. Dips mean it's waiting on something (usually data)."""
    mins = [s / 60 for s in g["elapsed_s"]]
    fig, ax = plt.subplots()
    ax.plot(mins, g["gpu_util_pct"], color=GREEN, linewidth=1.2)
    avg = sum(g["gpu_util_pct"]) / len(g["gpu_util_pct"])
    ax.axhline(avg, color=GRAY, linestyle="--", linewidth=1, label=f"avg {avg:.0f}%")
    ax.set_title("GPU utilization")
    ax.set_xlabel("elapsed minutes")
    ax.set_ylabel("GPU utilization (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    _save(fig, out_dir, "08_gpu_utilization.png")


def chart_gpu_power(g, out_dir):
    """Q: How hard is the GPU working in watts, and are we power-capped?
    Board power draw against the power limit. Riding the limit line means the
    card is maxed out (good for utilization). Also a proxy for energy cost."""
    mins = [s / 60 for s in g["elapsed_s"]]
    fig, ax = plt.subplots()
    ax.plot(mins, g["power_draw_w"], color=RED, linewidth=1.2, label="draw")
    if "power_limit_w" in g and isinstance(g["power_limit_w"][0], float):
        ax.axhline(g["power_limit_w"][0], color=GRAY, linestyle="--",
                   linewidth=1, label=f"limit {g['power_limit_w'][0]:.0f} W")
    ax.set_title("GPU power draw")
    ax.set_xlabel("elapsed minutes")
    ax.set_ylabel("watts")
    ax.legend()
    _save(fig, out_dir, "09_gpu_power.png")


def chart_gpu_temp(g, out_dir):
    """Q: Is the GPU thermally healthy?
    Temperature over time. Steady is fine; a climb toward ~83C+ risks thermal
    throttling, which would quietly drop throughput."""
    mins = [s / 60 for s in g["elapsed_s"]]
    fig, ax = plt.subplots()
    ax.plot(mins, g["temp_c"], color=ORANGE, linewidth=1.2)
    ax.set_title("GPU temperature")
    ax.set_xlabel("elapsed minutes")
    ax.set_ylabel("°C")
    _save(fig, out_dir, "10_gpu_temperature.png")


def chart_gpu_mem_smi(g, out_dir):
    """Q: How full is VRAM at the hardware level?
    Memory used vs total from nvidia-smi (includes everything on the card, not
    just the model's tensors). The gap to 'total' is your real headroom."""
    mins = [s / 60 for s in g["elapsed_s"]]
    used_gb = [v / 1024 for v in g["mem_used_mib"]]
    fig, ax = plt.subplots()
    ax.plot(mins, used_gb, color=PURPLE, linewidth=1.2, label="used")
    if "mem_total_mib" in g and isinstance(g["mem_total_mib"][0], float):
        total_gb = g["mem_total_mib"][0] / 1024
        ax.axhline(total_gb, color=GRAY, linestyle="--", linewidth=1,
                   label=f"total {total_gb:.0f} GB")
    ax.set_title("GPU memory (nvidia-smi, whole card)")
    ax.set_xlabel("elapsed minutes")
    ax.set_ylabel("GB")
    ax.legend()
    _save(fig, out_dir, "11_gpu_mem_smi.png")


# ----- Headline summary ----------------------------------------------------

def write_summary(m, g, out_dir):
    """Write a plain-text summary of the headline numbers for the deck's title
    slide, so you don't have to eyeball them off the charts."""
    lines = ["Solu-135M training run - headline numbers", "=" * 44, ""]

    if m:
        n = len(m["step"])
        tps = m["tokens_per_sec"]
        med_tps = sorted(tps)[len(tps) // 2]
        final_val = m["val_loss"][-1]
        best_val = min(m["val_loss"])
        final_ppl = m["val_perplexity"][-1]
        best_ppl = min(m["val_perplexity"])
        peak_mem = max(m["gpu_mem_gb"])
        lines += [
            f"logged points:        {n}",
            f"final val loss:       {final_val:.4f}",
            f"best val loss:        {best_val:.4f}",
            f"final val perplexity: {final_ppl:.2f}",
            f"best val perplexity:  {best_ppl:.2f}",
            f"median throughput:    {med_tps:,.0f} tokens/sec",
            f"peak model VRAM:      {peak_mem:.2f} GB (PyTorch max-allocated)",
            "",
        ]

    if g:
        avg_util = sum(g["gpu_util_pct"]) / len(g["gpu_util_pct"])
        avg_power = sum(g["power_draw_w"]) / len(g["power_draw_w"])
        max_temp = max(g["temp_c"])
        run_minutes = g["elapsed_s"][-1] / 60
        lines += [
            f"avg GPU utilization:  {avg_util:.0f}%",
            f"avg power draw:       {avg_power:.0f} W",
            f"max temperature:      {max_temp:.0f} C",
            f"GPU-logged duration:  {run_minutes:.1f} min",
            "",
        ]

    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {path}")
    print()
    print("\n".join(lines))


# ----- Main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics.csv")
    ap.add_argument("--gpu", default="gpu_metrics.csv")
    ap.add_argument("--out", default="graphs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    m = load_csv(args.metrics)
    g = load_csv(args.gpu)

    if not m:
        print(f"no training metrics found at {args.metrics} - nothing to graph")
    else:
        print(f"loaded {len(m['step'])} rows from {args.metrics}")
        chart_loss(m, args.out)
        chart_perplexity(m, args.out)
        chart_lr(m, args.out)
        chart_grad_norm(m, args.out)
        chart_throughput(m, args.out)
        chart_step_time(m, args.out)
        chart_gpu_mem_from_train(m, args.out)

    if not g:
        print(f"no GPU metrics found at {args.gpu} - skipping hardware charts "
              f"(run gpu_logger.py during training to get them)")
    else:
        print(f"loaded {len(g['elapsed_s'])} rows from {args.gpu}")
        chart_gpu_util(g, args.out)
        chart_gpu_power(g, args.out)
        chart_gpu_temp(g, args.out)
        chart_gpu_mem_smi(g, args.out)

    if m or g:
        write_summary(m, g, args.out)
        print(f"\ndone. charts are in {args.out}/")


if __name__ == "__main__":
    main()

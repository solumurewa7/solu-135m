"""
GPU-utilization logger for Solu-135M training.

Runs as its own process, separate from the training script, and polls
`nvidia-smi` on a fixed interval. Keeping it separate matters: querying the GPU
from inside the training loop would stall the loop and distort the timing we're
trying to measure. This process just watches from the side and writes a CSV.

The way this connects to the rest of the instrumentation is that train.py logs
per-step model metrics (loss, tokens/sec) while this logs hardware metrics
(utilization, power, temperature, memory). The grapher later reads both.

Run it in the background on the GPU box, started just before training:
    nohup python -u train/gpu_logger.py > gpu_logger.log 2>&1 &

Stop it when training ends:
    pkill -f gpu_logger.py
"""

import csv
import subprocess
import time
import os
from datetime import datetime, timezone

GPU_CSV = "gpu_metrics.csv"
POLL_SECONDS = 5   # how often to sample the GPU

# The fields we pull from nvidia-smi, in query order. Each maps to a CSV column.
# --query-gpu names come straight from `nvidia-smi --help-query-gpu`.
QUERY_FIELDS = [
    "utilization.gpu",       # % of time the GPU was doing work
    "utilization.memory",    # % of time the memory interface was busy
    "memory.used",           # MiB of VRAM in use
    "memory.total",          # MiB of VRAM total
    "power.draw",            # current board power in watts
    "power.limit",           # the power cap in watts
    "temperature.gpu",       # core temp in Celsius
    "clocks.current.sm",     # SM (compute) clock in MHz
]

CSV_COLUMNS = [
    "timestamp", "elapsed_s",
    "gpu_util_pct", "mem_util_pct",
    "mem_used_mib", "mem_total_mib",
    "power_draw_w", "power_limit_w",
    "temp_c", "sm_clock_mhz",
]


def sample_gpu():
    """Return one reading from nvidia-smi as a list of strings, or None on error."""
    query = ",".join(QUERY_FIELDS)
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"nvidia-smi query failed: {e}")
        return None
    # One line per GPU; this project trains on a single GPU, so take the first.
    first_line = out.strip().splitlines()[0]
    return [v.strip() for v in first_line.split(",")]


def main():
    is_new = not os.path.exists(GPU_CSV)
    f = open(GPU_CSV, "a", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(CSV_COLUMNS)
        f.flush()

    print(f"logging GPU every {POLL_SECONDS}s to {GPU_CSV} (Ctrl-C / pkill to stop)")
    start = time.time()
    try:
        while True:
            values = sample_gpu()
            if values is not None:
                ts = datetime.now(timezone.utc).isoformat()
                elapsed = round(time.time() - start, 1)
                writer.writerow([ts, elapsed] + values)
                f.flush()   # flush each row so a kill doesn't lose data
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nstopping GPU logger")
    finally:
        f.close()


if __name__ == "__main__":
    main()

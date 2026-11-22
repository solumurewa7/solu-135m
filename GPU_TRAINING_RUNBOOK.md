# Solu-135M — GPU training run (runbook)

This is the exact sequence for the real training run on an A100, with the full
instrumentation suite logging the whole time. Follow it top to bottom. The point
of writing it out is that A100 time costs money, so you don't want to be figuring
out commands while the meter runs.

Prereqs: the 10B shards are already built and sitting on your **Network Volume**
(from BUILD_10B_RUNBOOK.md). The volume is the thing that persists; pods are
disposable.

---

## 1. Deploy the GPU pod and attach the volume

- Runpod -> deploy a **GPU pod: A100 SXM 80GB**.
- **Attach the same Network Volume** you built the shards on, mounted at
  `/workspace`. It must be the same volume, in the same region (EU-RO-1), or the
  shards won't be there.
- Pick a PyTorch / CUDA template. Open the web terminal.

Confirm the GPU and the shards are both really there before spending time:
```bash
nvidia-smi                       # should show one A100 80GB
ls -lh /workspace/solu-135m/shards/ | head
```
You want to see ~100 shard files. If `shards/` is empty, you attached the wrong
volume or the wrong region. Stop and fix that first.

## 2. Get the latest code and install deps

```bash
cd /workspace/solu-135m
git pull                         # pull the instrumentation you just pushed
pip install -r requirements.txt
```
`matplotlib` will install too; that's fine, it's small. You just won't run the
grapher here (that's the laptop's job, step 7).

## 3. Sanity-check that the pipeline sees the shards

```bash
python -c "from train.data_pipeline import load_shards; tr, va = load_shards(); print(len(tr), 'train shards,', len(va), 'val shards')"
```
This should print your shard counts in a second or two. Because the shards
already exist, `build_shards()` skips rebuilding. If this hangs or tries to
download, something's wrong with the volume, not the code.

## 3b. Run the smoke test (do this once, ideally on the CPU box)

Before spending GPU time, prove the whole training loop runs end to end against
the real shards. This is cheap insurance: it catches a dumb bug (bad shape, bad
CSV column, dtype mismatch) in a minute on CPU instead of after you've rented an
A100 and waited through torch.compile.

```bash
python train/smoke_test_train.py
```
It runs ~10 real steps with a tiny model on CPU and should end with
`SMOKE TEST PASSED`. The loss will sit around 10.8 and not drop; that's correct
(a toy model on a few steps learns nothing, it only proves the machinery works).
If it raises, fix that before touching a GPU. If your shards live somewhere other
than `shards/`, set `SOLU_SHARD_DIR` to that folder first.

> This is ideal to run on the CPU tokenizing box the moment the shards finish,
> while that box is still up, since it already has transformers/datasets
> installed and the shards right there.

## 4. Start the GPU logger FIRST, then training

Order matters: start the hardware logger a moment before training so it captures
the whole run, including warmup.

```bash
# hardware metrics -> gpu_metrics.csv, every 5s, in the background
nohup python -u train/gpu_logger.py > gpu_logger.log 2>&1 &

# the training run -> metrics.csv + TensorBoard, also in the background
nohup python -u train/train.py > train.log 2>&1 &
```

`nohup ... &` means both survive if your terminal disconnects. The logger keeps
running until you kill it in step 6.

## 5. Watch the first ~30 steps closely (the money check)

```bash
tail -f train.log
```

Two things to confirm in the first minute or two:

1. **No OOM.** Batch 32 is a calculated guess (we measured 12 GB at batch 8 and
   extrapolated), not a measured value, so the first real steps are the test. If
   you see `CUDA out of memory`, kill training, open `train/data_pipeline.py`,
   set `BATCH_SIZE = 16`, raise `TOTAL_STEPS` to ~610000 in `train/train.py`, and
   restart from step 4. Don't just lower it blindly if it fits; 32 is faster.
2. **Throughput is in range.** The per-step log line prints `tok/s`. The first
   few steps are slow (torch.compile is building kernels). After ~20-30 steps it
   should settle near ~100,000 tok/s. If it settles far below that, something is
   bottlenecked and it's worth stopping to look before paying for the full run.

The log line looks like:
```
step 30: train 7.21 | val 7.44 | lr 0.000045 | gnorm 0.63 | 101,340 tok/s | 11.6 GB
```
Once you've seen a healthy line like that, you can stop watching. It'll run for
~27 hours of GPU time (10B tokens / ~100k tok/s). Checkpoints save to the volume
every ~11 min, so a crash loses at most that.

Optional, from a second terminal, live graphs in the browser:
```bash
tensorboard --logdir runs --port 6006
```
(Then open the pod's port 6006.) TensorBoard is just for watching; the CSVs are
the real record.

## 6. When training finishes

```bash
pkill -f gpu_logger.py           # stop the hardware logger
ls -lh metrics.csv gpu_metrics.csv best_checkpoint.pt checkpoint.pt
```
You now have four things that matter:
- `metrics.csv` — per-step model metrics
- `gpu_metrics.csv` — per-5s hardware metrics
- `best_checkpoint.pt` — lowest val-loss weights (the ones to keep)
- `checkpoint.pt` — rolling checkpoint (for resuming)

## 7. Pull the data down and make the graphs (on your laptop)

Download the two CSVs and the best checkpoint off the volume (Runpod file browser
or `scp`). Then, locally:
```bash
python train/make_graphs.py --metrics metrics.csv --gpu gpu_metrics.csv --out graphs
```
That writes 11 labeled PNGs plus `summary.txt` into `graphs/`. Those are the
deck. The rule we've followed the whole time: log cheap on the GPU, draw the
pictures later on the laptop, so you never pay A100 rates to render a chart.

## 8. Release the GPU, keep the volume

- **Terminate the GPU pod** (stops the expensive billing).
- **Keep the Network Volume** (holds your checkpoints and CSVs, cheap monthly
  storage). Don't delete it until the weights are safely downloaded and, ideally,
  pushed to the HF Hub.

---

## Cost recap (expected)

- Throughput: ~100,000 tok/s on the A100.
- 10B tokens: ~27-28 GPU-hours. At A100 SXM pricing that's roughly $60 ideal,
  budget $70-90 with warmup, checkpoint stalls, and the sanity-check time.
- Plus the CPU tokenizing box (a few dollars, already spent) and small monthly
  volume storage.

## If something breaks mid-run

- **Pod dies / you disconnect:** the volume kept the checkpoints. Deploy a fresh
  A100, attach the same volume, `git pull`, and just run step 4 again. `train.py`
  finds `checkpoint.pt` and resumes from the last saved step. The logger starts a
  fresh `gpu_metrics.csv` append, which is fine.
- **OOM after some steps (not step 0):** rare, but if it happens deep in the run,
  it's usually a long sequence. Lower `BATCH_SIZE` and resume; the checkpoint
  still loads.
- **Throughput slowly drops over hours:** check `gpu_metrics.csv` temperature
  later; sustained ~83C+ means thermal throttling, which is the host's problem,
  not yours. Usually not an issue on datacenter A100s.

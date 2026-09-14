# SLURM scripts

The scripts in this folder are for **GAIVI** (USF). For **HiPerGator** (UF), use
`hipergator/` instead — different partitions, conda setup and storage rules; see
`hipergator/README.md`.

**Create `logs/` before submitting** (`mkdir -p logs` in `analysis/`). SLURM opens
`--output=logs/...` when the job *starts*, before the script's own `mkdir -p logs`
runs, so on a fresh checkout the first job's log has nowhere to go.

Submit from `analysis/` (the scripts `cd "$SLURM_SUBMIT_DIR"` and then into `src/`):

```bash
cd analysis
sbatch slurm/gen_data.sh                     # CPU only, ONE arm (pile-up) + normalisation
sbatch slurm/train_diffusion.sh              # EDM
sbatch slurm/train_wgan.sh                   # Morovati baseline
```

Order matters: `gen_data.sh` must finish before either training job. Chain them so
you do not have to babysit:

```bash
JID=$(sbatch --parsable slurm/gen_data.sh)
sbatch --dependency=afterok:$JID slurm/train_diffusion.sh
sbatch --dependency=afterok:$JID slurm/train_wgan.sh
```

When both have finished, one table — same test patches, same metric code for both:

```bash
cd src && python3 compare_baseline.py --arm baseline3d_pu_matched
```

**Current plan is one arm**, `baseline3d_pu_matched`: tau = 30 ns paralyzable pile-up,
because Morovati et al.'s input includes pile-up. Both models train and test on it
(14 / 3 / 3 phantoms). The second arm, for the mismatch experiment later, is the same
script: `sbatch --export=ALL,TAU=0,ARM=baseline3d_matched slurm/gen_data.sh`.

Override defaults without editing the files:

```bash
sbatch --export=ALL,ARM=baseline3d_pu_matched,ITERS=100000 slurm/train_diffusion.sh
```

## If a job hits the time limit

Both trainers take `--resume` (the scripts pass it by default). Checkpoints are written
every 10k iterations, atomically, with optimizer state. If a job is killed at 22 h,
**resubmit the same script** — it continues from the last checkpoint instead of
restarting. Evaluation / coverage only runs after training reaches its final iteration.

This matters mostly for the WGAN: `train_wgan.sh` now uses `--epochs 40` to match
Morovati et al.'s schedule, which is ~1.09M iterations on our data at batch 64 — about
5.4x the old 200k default, and possibly more than one allocation. Verified on CPU:
a 14-iteration run resumed at 14 and finished at 28.

## Deliberate differences from the template

**`--gres=gpu:1`, not `gpu:2`.** Neither `train.py` is distributed — no DDP, no
DataParallel — so a second GPU sits idle and only lengthens the queue wait. Ask for 2
only after adding DDP.

**`gen_data.sh` requests no GPU at all.** Dataset generation is pure numpy. Putting it
in the GPU queue would waste an allocation for ~10 hours.

**`--cpus-per-task` raised to 4-8.** `gen_data.sh` is numpy-bound and benefits from
BLAS threads; the training jobs need CPU for the patch loader feeding the GPU.

**Smoke test runs first in `train_diffusion.sh`,** and the job exits if it fails. The
UNet, sampler and training loop have never been executed — no torch on the dev machine —
so the first GPU run is where shape bugs surface. One minute of checking beats losing a
22-hour allocation to a crash at iteration 3.

**`--output` / `--error` into `logs/`** with the job id, so concurrent runs do not
overwrite each other.

## Check before the first real submission

- **`-p Quick` with `--time=22:00:00`.** If `Quick` is a short-queue partition, this will
  be rejected or silently truncated. Confirm with `sinfo -o "%P %l"` and move to a longer
  partition if needed.
- **`pip install -r requirements.txt` runs every job.** Fine the first time; afterwards
  it is a slow no-op you can comment out. `requirements.txt` does NOT pin torch to a CUDA
  build — install the right one into the `proj` env once, by hand.
- **Runtime:** generation ~5-6 h per arm (20 phantoms), so `gen_data.sh` needs its full
  22 h for both. Training time is unmeasured — nothing has run on a GPU yet.

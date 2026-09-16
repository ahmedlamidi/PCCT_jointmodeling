# Runbook — WGAN vs diffusion, one arm, on HiPerGator

Do the steps in order. Each has a **checkpoint**: the output that means it worked.
Stop at the first checkpoint that does not match and fix it before going on.

Fill in once:  `<gatorlink>` = your UF username,  `<group>` = your HiPerGator group
(`id -gn` on HiPerGator usually gives it; step 4 confirms).

Nothing below has run on HiPerGator yet. It was written from the UFRC docs and
tested locally as far as possible without cluster access — see
`slurm/hipergator/README.md` for the sources and what is unverified.

---

## A · On your machine

- [ ] **1. Commit the HiPerGator scripts** (they are untracked).
  ```bash
  cd PCCT_jointmodeling
  git add .gitignore analysis/RUNBOOK.md analysis/slurm/
  git commit -m "HiPerGator job scripts and runbook"
  ```
  Checkpoint: `git status --short` prints nothing for `analysis/slurm/`.

- [ ] **2. (Optional) Reclaim 2.4 GB of dead git objects.** Deletes only objects no
  commit uses; permanent.
  ```bash
  git gc --prune=now && git count-objects -vH
  ```
  Checkpoint: `size` drops from ~2.45 GiB to a few MB.

Layout: the git clone lives in **home** (`~/PCCT_jointmodeling`); everything jobs
*write* (dataset ~15 GB per arm, checkpoints, logs) lives on **/blue** through two
symlinks. Home is 40 GB, so `analysis/outputs` must not be a real directory there.

- [ ] **3a. Clone on HiPerGator** (done: `/home/<gatorlink>/PCCT_jointmodeling`).

- [ ] **3b. Point outputs and logs at /blue** — on HiPerGator, BEFORE running anything
  (any script import creates `analysis/outputs` as a real directory in home):
  ```bash
  B=/blue/<group>/<gatorlink>/pcct
  mkdir -p $B/outputs $B/logs
  cd ~/PCCT_jointmodeling/analysis
  ln -s $B/outputs outputs
  ln -s $B/logs logs
  ls -l outputs logs
  ```
  Checkpoint: both lines show `-> /blue/...`. If `ln` says *File exists*, a real
  directory got there first — `rmdir outputs` (it is empty) and redo the `ln`.

- [ ] **3c. Send the gitignored inputs** (~430 MB, 17 files). From `PCCT_jointmodeling/`
  on your machine:
  ```bash
  rsync -avPR --exclude '.DS_Store' \
    analysis/RUNBOOK.md \
    analysis/cache \
    PcTK_3.24a/1_inputdata \
    PcTK_3.24a/5_refdata/dat_nCovE_ver3.2_dpix_225_dz_1600_r0_24_esig_2.0.mat \
    <gatorlink>@hpg.rc.ufl.edu:/home/<gatorlink>/PCCT_jointmodeling/
  ```
  Checkpoint: ~430 MB sent. PcTK is licensed (no redistribution) — it stays in
  your home, never in a group-readable directory.

## B · On a HiPerGator login node

```bash
ssh <gatorlink>@hpg.rc.ufl.edu
cd ~/PCCT_jointmodeling/analysis
```

- [ ] **4. Confirm your account and QOS.**
  ```bash
  module load ufrc && slurmInfo
  ```
  Checkpoint: your group is listed, with GPU resources. If it differs from
  `id -gn`, prefix every later command with `GROUP=<that name>`.

- [ ] **5. Build the environment** (once, ~10 min).
  ```bash
  bash slurm/hipergator/setup_env.sh
  ```
  Checkpoint: last lines read `torch 2.x | CUDA build 12.8` and
  `environment ready: /blue/...`. If `CUDA build` is `None`, torch is a CPU
  build — rerun the `pip install torch --index-url .../cu128` line.

- [ ] **6. Preflight.**
  ```bash
  bash slurm/hipergator/preflight.sh
  ```
  Checkpoint: ends with `ALL CHECKS PASSED`. Every `sbatch --test-only` line should
  say the job *would* start — the scheduler has validated partition, account, QOS
  and resources without running anything.

## C · Submit and watch

- [ ] **7. Submit everything.**
  ```bash
  bash slurm/hipergator/submit.sh              # or: QOS=burst bash slurm/hipergator/submit.sh
  ```
  Checkpoint: prints three job ids — `gendata=... edm=... wgan=...`.
  `squeue -u $USER` shows the data job running and the two training jobs
  `PENDING (Dependency)`.

- [ ] **8. Data job, ~6 h expected** — `tail -f logs/gendata_<id>.out`.
  Checkpoints, in order:
  ```
  phantoms: 14 train / 3 val / 3 test  (20 total, 70/15/15%)
  label=clean  bins=9  (closed top bin, 100-109 keV)
  detector plane 1854 ch x 32 rows,  180 views
  patches: 690 per projection, ...
  pileup grid brain   0.0 cm ...        <- 24 of these, ~25 min total
    train  1  soft<=... cm bone<=... cm  X(180, 32, 1854, 9) ...
  ...
  wrote .../outputs/baseline3d_pu_matched
  normalisation defined by arm: baseline3d_pu_matched
  ```
  Then confirm the physical ceiling was written:
  ```bash
  python -c "import json; print(json.load(open('outputs/baseline3d_pu_matched/manifest.json'))['air_counts'])"
  ```
  Nine numbers of order 200–2200.

- [ ] **9. Diffusion job** — `tail -f logs/edm_<id>.out`. Checkpoints:
  - `nvidia-smi -L` names an L4
  - `ALL CHECKS PASSED` from the smoke test (if not, the job stops — send the log)
  - `preloaded 14 phantoms (10.8 GB) in ... s` — all training data now in memory
  - `it  200  loss ...` lines, with loss drifting below ~1.0 as it learns
  - ends with `wrote .../outputs/coverage_baseline3d_pu_matched_Y.json`

- [ ] **10. WGAN job** — `tail -f logs/wgan_<id>.out`. Checkpoints:
  - `40 epochs over 1738800 train patches at batch 64 -> 1086750 iterations`
  - `preloaded 14 phantoms (10.8 GB) in ... s` — all training data now in memory
  - `it ... mse ... rmae ...` lines, with MSE falling
  - ends with `wrote .../outputs/wgan_eval_baseline3d_pu_matched.json`
  - **If it stops with TIMEOUT instead**, resubmit — it resumes from its last checkpoint:
    ```bash
    sbatch --account=<group> --qos=<group> slurm/hipergator/train_wgan.sh
    ```
    Repeat until the evaluation line appears. The same works for the diffusion job.
  - **Started before the 2026-09-14 loader fix?** A log with no `preloaded` line that
    runs at ~2.8 it/s is spending ~90% of its time re-decompressing phantom files
    (measured). Commit and push the fix on your machine, `git pull` in
    `~/PCCT_jointmodeling` on HiPerGator, then `scancel <id>` and resubmit the same
    script. It resumes from its last checkpoint (saved every 10k iterations), so at
    most ~10k iterations are redone.

- [ ] **10b. Windowed SSIM + sinogram error figure** (after step 9; GPU, hours).
  The SSIM in the step-9 log is one global window and reads ~0.9999 for anything.
  ```bash
  sbatch --account=<group> --qos=<group> slurm/hipergator/ssim_diffusion.sh
  ```
  Checkpoints in `logs/ssim_<id>.out`, earliest result first:
  - `preview: phantom 0 with 2 samples per patch -> .../figures/preview` then
    `preview: sinogram RMSE ... -- PREVIEW only` — a first look in ~1/8 of a phantom's time.
    Figures in `figures/preview/`, data in `outputs/ssim_..._preview/`. Not a result: no
    uncertainty band, never read by step 11. `PREVIEW_NSAMP=0` in `--export` turns it off.
  - `chunk 1/21 ... eta N min` — phantom 0 in full, the figure phantom; the eta is its real time
  - `wrote .../figures/sino_error_*.png` and `figures/profiles_*_bin{1,5,9}.png` — the
    sinogram and **time-series** figures, ready as soon as phantom 0 is done
  - `[saved after figure phantom 0 ...]`, then the patch domain:
    `posterior-mean RMSE ... (coverage.py: 4.2591 ...)` — the two agree to a few decimals
  - phantoms 1 and 2, each followed by `[saved after ...]`
  - ends with `wrote .../outputs/ssim_baseline3d_pu_matched_Y_on_baseline3d_pu_matched.json`
  The JSON is rewritten after every step (`"complete": false` until the last phantom), so
  a TIMEOUT keeps everything finished; resubmit the same line and it continues. Step 11
  can run before it ends — its SSIM rows then read e.g. `0.8123 (1/3 ph.)`.
  For another channel, view or bin — no GPU, also works on a copied-home slice file:
  ```bash
  cd src/diffusion && python plot_profiles.py \
    ../../outputs/ssim_baseline3d_pu_matched_Y_on_baseline3d_pu_matched/slice_ph000_row16.npz \
    --bin 0 --channel 1200 --view 45
  ```
  Every plot as its own figure (no overviews) — also no GPU; add `--preview` for a
  `..._preview/` slice file (no std figure, no band):
  ```bash
  cd src/diffusion && python plot_singles.py <slice_ph000_row16.npz> --out <folder>
  ```
  Uncertainty map (posterior std), error map and |error|/std map of the whole sinogram, per bin:
  ```bash
  cd src/diffusion && python plot_maps.py <slice_ph000_row16.npz> --out <folder>            # diffusion
  cd src/diffusion && python plot_maps.py <ssim_wgan_.../slice_ph000_row16.npz> --out <folder> --model_label WGAN
  ```
  Does the uncertainty track the error? Per-bin Spearman correlation and reliability curves
  (pixels grouped by predicted std vs their actual RMSE; `--region head|air|all`):
  ```bash
  cd src/diffusion && python plot_unc_vs_err.py <slice_ph000_row16.npz> --out <folder>
  ```
  After step 10 (WGAN trained), score the WGAN the same way — one pass per patch, much faster:
  ```bash
  sbatch --account=<group> --qos=<group> --export=ALL,MODEL=wgan slurm/hipergator/ssim_diffusion.sh
  ```
  Whichever of the two finishes second also writes `figures/profiles_compare_*_bin{1,5,9}.png`
  (input, label, diffusion and WGAN on one sinogram trace), and step 11's table gains
  the windowed-SSIM rows.

## D · The result

- [ ] **11. Compare.**
  ```bash
  cd src && python compare_baseline.py
  ```
  Checkpoint: one table — RMSE, PSNR, SSIM, manifold residual, coverage. If it
  says the evaluation sets differ, one job was run with a different `--npatch`.

- [ ] **12. Bring the results home.** From your machine:
  ```bash
  rsync -avP --include='compare_*' --include='coverage_*' --include='wgan_eval_*' --include='ssim_*.json' \
    --exclude='*' <gatorlink>@hpg.rc.ufl.edu:/blue/<group>/<gatorlink>/pcct/outputs/ analysis/outputs/
  ```
  Training curves — one CSV per model, one row per 200 iterations (`src/trainlog.py`;
  runs started or resumed after 2026-09-14 only — earlier ones have just the SLURM logs):
  ```bash
  rsync -avP --prune-empty-dirs --include='edm_*/' --include='wgan_*/' --include='train_log.csv*' \
    --exclude='*' <gatorlink>@hpg.rc.ufl.edu:/blue/<group>/<gatorlink>/pcct/outputs/ analysis/outputs/
  rsync -avP --include='edm_*' --include='wgan_*' --exclude='*' \
    <gatorlink>@hpg.rc.ufl.edu:/blue/<group>/<gatorlink>/pcct/logs/ analysis/logs/
  ```

How to read the table: expect the WGAN to win RMSE/PSNR — its loss weights MSE at
1000. Coverage is the row only the diffusion model can fill.

---

## If something fails

| Symptom | Cause | Fix |
|---|---|---|
| `Invalid account or account/partition combination` | group name wrong | step 4, then `GROUP=<name>` |
| `Invalid qos specification` | QOS not yours | `QOS=burst` or check `slurmInfo` |
| job pending for hours | group's GPUs busy | `QOS=burst`, or `--partition=hpg-b200` |
| `FileNotFoundError ... PcTK_3.24a` | step 3 incomplete | rerun the rsync |
| smoke test prints `device cpu`, or training is very slow | CPU-only torch | step 5 |
| `oom-kill` in the data job | memory | resubmit with `--mem=96gb` |
| `TIMEOUT` in a training job | expected for 40 epochs | resubmit the same script (step 10) |
| smoke test fails | a real bug | stop, send `logs/edm_<id>.out` |

## After this works

The mismatch experiment needs the second arm — same script, pile-up off:
```bash
sbatch --account=<group> --qos=<group> --export=ALL,TAU=0,ARM=baseline3d_matched \
       slurm/hipergator/gen_data.sh
```
Then evaluate the already-trained models on it. Do **not** rebuild the
normalisation from that arm.

---

## E · Morovati et al.'s own problem (separate arm, optional)

Their phantom, not ours: a water sphere with five ellipsoids of different tissues
(soft tissue, adipose, grey/white matter, blood, cortical bone), 256³ voxels at
0.113 mm, 10 train + 5 test, NIST ICRU-44 attenuation. Every matching choice is a
default of `src/generate_morovati_dataset.py`; its docstring lists what is matched,
what is not, and why. Independent of steps 1–12 — it writes only
`outputs/morovati_match/` and never touches `outputs/norm_stats.json`.

- [ ] **13. Generate** (CPU, ~4 h: ~4 min pile-up grid, then ~15 min per phantom —
  measured locally at 4 views and scaled to 180; peak memory 1.7 GB at 4 views, a
  few GB at 180).
  ```bash
  sbatch --account=<group> --qos=<group> slurm/hipergator/gen_morovati.sh
  ```
  Checkpoints in `logs/morovati_<id>.out`:
  ```
  phantoms: 10 train / 0 val / 5 test
  materials as a*brain + b*cortical bone (NIST ICRU-44, fit 20-120 keV):   <- six lines, max err <= 0.84%
  object: 256^3 voxels x 0.113 mm = 28.9 mm cube, sphere 26.0 mm across
  pile-up grid: brain-eq 0.00..2.75 cm (12)  x  bone-eq -0.11..2.75 cm (13) = 156 points
  detector plane 272 ch x 256 rows, 180 views over 180 deg
  patches: 1023 per projection, 2762100 total   (theirs: 1023 / 1,841,400 for 10 phantoms)
    train  1  adipose/blood/...  brain-eq<=2.6 cm  bone-eq ...  X(180, 256, 272, 9)  ...s
  ...
  normalisation NOT built: outputs/norm_stats.json belongs to the head arm.
  ```
  1023 per projection is theirs exactly; the total is 1.5x theirs only because it
  counts the 5 test phantoms too. A `path outside the pile-up grid` stop means the
  phantom settings were changed without the grid following — send the log.

Training on this arm is not wired up yet: it needs its own normalisation file
first (the one in `outputs/` belongs to the head arm and must not be overwritten).

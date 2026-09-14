# Running on HiPerGator

The one-arm WGAN-vs-diffusion comparison, adapted to HiPerGator. Every setting
below comes from UFRC's own documentation, linked inline.

## Why these differ from the GAIVI scripts

| | GAIVI template | HiPerGator |
|---|---|---|
| CPU partition | `-p Quick` | `hpg-default`, 31-day cap ([partition limits](http://docs.rc.ufl.edu/scheduler/partition_limits/)) |
| GPU request | `--gres=gpu:2` | `--partition=hpg-turin --gpus=1` (L4); also `hpg-rtx6000`, `hpg-b200`; 14-day cap; ≥1 CPU per GPU ([GPU access](http://docs.rc.ufl.edu/scheduler/gpu_access/)) |
| Account / QOS | none | investment QOS = your group, 744 h; burst = `<group>-b`, 96 h ([QOS limits](http://docs.rc.ufl.edu/scheduler/qos_limits/)) |
| Conda | `source /apps/anaconda3/...` | `module load conda`; envs on `/blue` ([conda config](https://docs.rc.ufl.edu/software/conda_configuration)) |
| Torch | `pip install` every job | once: `pip install torch --index-url https://download.pytorch.org/whl/cu128` — covers L4 (CUDA ≥ 12.0) and B200 (≥ 12.8.1) ([UFRC conda](http://docs.rc.ufl.edu/software/conda_installing_packages/)) |
| Storage | — | home is 40 GB; **all job I/O belongs on `/blue`** ([storage](https://docs.rc.ufl.edu/quickstart/practical_storage/)). This writes ~20 GB |

## 1 · Copy the project to /blue

Git will **not** carry everything: the PcTK tables and `analysis/cache/` are
gitignored, and the data job crashes without them. From the `PCCT_jointmodeling/`
root on your machine:

```bash
rsync -avP \
  --exclude '.git/' \
  --exclude 'analysis/outputs/' --exclude 'analysis/figures/' --exclude 'analysis/logs/' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude 'PcTK_3.24a.zip' \
  --exclude 'PcTK_3.24a/2_outputdata/' --exclude 'PcTK_3.24a/3_src/' --exclude 'PcTK_3.24a/4_doc/' \
  --include 'PcTK_3.24a/5_refdata/dat_nCovE_ver3.2_dpix_225_dz_1600_r0_24_esig_2.0.mat' \
  --exclude 'PcTK_3.24a/5_refdata/*' \
  ./ <gatorlink>@hpg.rc.ufl.edu:/blue/<group>/<gatorlink>/PCCT_jointmodeling/
```

**430 MB, 108 files** (checked with a local dry run): all code, PcTK's `1_inputdata`,
the one covariance table the code reads from `5_refdata`, and `analysis/cache`.
Keep the `--include` line *before* the `5_refdata/*` exclude — rsync applies the
first rule that matches.

What it deliberately leaves out:
- **`.git/`** — 2.45 GiB of loose objects, most of them unreachable (leftovers from
  an earlier `git add` of large files). Nothing on HiPerGator needs git history.
- **The rest of `5_refdata`** — reference sinograms and images the code never reads.
- **Your outputs and figures.**

PcTK is licensed for your own use — this copies it to your own allocation, it does
not redistribute it.

## 2 · One-time environment (login node)

```bash
cd /blue/<group>/<gatorlink>/PCCT_jointmodeling/analysis
bash slurm/hipergator/setup_env.sh
```

## 3 · Preflight (login node) — do not skip

```bash
bash slurm/hipergator/preflight.sh
```

It checks the repo is on `/blue`, the gitignored inputs arrived, your account and
QOS (`slurmInfo`), the environment imports torch, and finishes with
`sbatch --test-only` on all three jobs — the scheduler validates partition,
account, QOS and resources **without running anything**. It must end with
`ALL CHECKS PASSED`.

## 4 · Submit

```bash
bash slurm/hipergator/submit.sh                # investment QOS
QOS=burst bash slurm/hipergator/submit.sh      # burst QOS, if your group's is busy
```

Chains the jobs: data (CPU) → EDM and WGAN (GPU, in parallel). Watch with
`squeue -u $USER`; logs in `analysis/logs/`.

If a training job hits its 72 h limit, resubmit just that script:
`sbatch --account=<group> --qos=<group> slurm/hipergator/train_wgan.sh`.
It resumes from the last checkpoint. The WGAN's 40 epochs (~1.09 M iterations)
is the one most likely to need this.

## 5 · Compare

```bash
cd src && python compare_baseline.py
```

## Not verified

These scripts have **never run on HiPerGator** — I have no access from here.
They were written against the UFRC docs, syntax-checked, and the preflight was
run locally, where it correctly fails the HiPerGator-only checks. Specifically
unconfirmed:

- **Account = your primary Unix group** (`id -gn`). Usual on HiPerGator, but
  preflight's `slurmInfo` output is the proof. Override with `GROUP=...`.
- **Compute-node internet.** UFRC doesn't document it, which is why nothing is
  installed inside a job.
- **Login host** `hpg.rc.ufl.edu` in the rsync line.
- **Runtimes on L4.** The 72 h limits are generous guesses; resubmission covers
  the case where they are not.

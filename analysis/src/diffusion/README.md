# Conditional EDM for PCCT — bin-count correction

Builds directly on Morovati et al. 2025 ([doi 10.1088/1361-6560/adaf71](https://iopscience.iop.org/article/10.1088/1361-6560/adaf71)):
**distorted energy-bin counts in, ideal energy-bin counts out**, patched on the
detector plane. Same task, same detector simulator (PcTK 3.2), same patching.
The addition is a *calibrated posterior* instead of a point estimate, and a test
of what that calibration does under forward-model mismatch.

Diffusion follows EDM — Karras, Aittala, Aila & Laine, NeurIPS 2022
([arXiv:2206.00364](https://arxiv.org/abs/2206.00364)).

```
norm_stats.py   fixed normalisation, built ONCE from the baseline arm
data.py         patch loader -> (condition, target)
unet.py         small UNet for 16x16 patches
edm.py          preconditioning (Table 1), loss, Heun sampler (Algorithm 2)
train.py        training loop with EMA
manifold.py     the 2-D surface physical count vectors live on + joint coverage
coverage.py     matched vs mismatched arm, three calibration measures
stitch.py       blend overlapping patches back into a full detector plane
smoke_test.py   run FIRST on the GPU box; catches shape bugs in ~1 min
```

```bash
python3 norm_stats.py --arm baseline3d_matched     # statistics from TRAIN only
python3 smoke_test.py
python3 train.py --arm baseline3d_matched --target Y --iters 200000 --batch 128
python3 coverage.py --ckpt ../../outputs/edm_baseline3d_matched_Y/ckpt.pt \
                    --arms baseline3d_matched baseline3d_pu_matched --nsamp 256
```

---

## The 2-D manifold — the reason this needs a generative model

Every noise-free bin count is a function of two numbers, cm soft tissue and cm
bone. So the 9-vectors do not fill 9-D space. Measured on 1854 rays:

```
component 1  91.58%
component 2   8.39%   (cumulative 99.97%)
component 3   0.03%
```

Three consequences:

1. A **diagonal-Gaussian head cannot represent this.** That is the concrete
   architectural argument, stronger than "generative models are expressive".
2. **Per-bin coverage cannot detect a violation.** A model can hit 90% in all
   nine bins while every sample sits off the surface — a count vector no
   attenuation path could produce.
3. It gives a validation test most uncertainty papers cannot run: project the
   samples onto the surface and measure the residual. Measured separation —
   true clean vectors 0.040, vs 0.145 once 5% independent per-bin noise is added.

`coverage.py` therefore reports per-bin coverage, joint coverage on the manifold,
and the manifold residual.

## Read the difference between arms, not the absolute number

`joint_coverage` uses an empirical sample quantile, which is optimistic with few
samples. A **perfectly calibrated** model still reads low:

```
nsamp   0.50   0.80   0.90   0.95   0.99
   64  0.500  0.776  0.866  0.921  0.967
  256  0.497  0.782  0.886  0.939  0.985
```

At nsamp=64 the 0.90 level is ~3.4 points low before the model does anything
wrong. The bias cancels between two arms at equal nsamp. Use 256+ for any
absolute claim, and never quote 0.99 at 64.

---

## Decided, and why

**Target = bin counts (`--target Y`).** Matches Morovati exactly, so numbers are
comparable, and it isolates the detector-model error from the decomposition
error. Materials can be derived afterwards — decomposition from 9 bins is a
quadratic with R^2 = 0.99999. `--target V` predicts materials directly if wanted.

**`sigma_data = 1.0`**, true by construction — `norm_stats.py` standardises the
target to unit variance; measured 1.028. Not EDM's 0.5, which assumes [-1,1] images.

**Normalisation frozen to `baseline3d`.** Per-arm statistics would make the
coverage comparison measure normalisation drift instead of model mismatch.

**Conditioning by channel concatenation**, not scaled by `c_in` (that
preconditions the noisy latent only). Input is `tgt_ch + 9` channels.

**Deterministic sampler (`S_churn = 0`).** Diversity comes from the initial
latent, so repeats stay reproducible.

**Counts clipped to [0, air count per bin]** after `expm1`, in `DiffusionPatches.denorm_t`, which the WGAN evaluation uses too. The upper clip is physical — no ray records more than the unattenuated beam — and necessary: one tail sample at +5 sigma in log space is ~1e6 counts and swamps an arithmetic posterior mean.

**Patches overlap and are blended**, not tiled. See `stitch.py`: plain averaging
leaves a gradient spike at every patch boundary (+0.0998 seam excess); a Hann
taper removes it (-0.0181) at essentially unchanged RMS. It fixes the *structure*
of the error, not its size.

## Splits

Phantom-level 70/15/15 train/val/test, set at generation time by
`--nphantom 20 --split 70 15 15`. The split is by PHANTOM, never by patch: patches
from one phantom are strongly correlated, so a patch-level split leaks and inflates
validation scores.

`DiffusionPatches(arm, split='val')` reads the new split with no change; `norm_stats.py`
draws from **train only**, so validation and test never touch the normalisation.

Note this departs from Morovati et al., who use 10 train + 5 test and no validation
split. `--ntrain 10 --ntest 5` reproduces theirs exactly if a like-for-like comparison
is wanted.

## Blocking, before any coverage number means anything

The datasets on disk still use the OLD label — a noisy ideal realisation sharing
its seed with the input. Morovati's label is **noise-free**. Regenerate:

```bash
python3 ../generate_baseline_dataset3d.py --label clean --closed_top_bin \
        --ntrain 10 --ntest 5 --out baseline3d
python3 ../generate_baseline_dataset3d.py --label clean --closed_top_bin \
        --tau 30 --T 25 --ntrain 10 --ntest 5 --out baseline3d_pu
```

`--closed_top_bin` also matches their nine CLOSED bins (20-29 ... 100-109). Our
open 100-inf bin collects the pile-up sum events their 110 keV threshold discards
— which is where the +943% air signal lives.

## Status (updated 2026-09-14)

**Executed end to end on CPU** (torch 2.14, isolated venv) on a tiny 4-phantom clean
arm: `smoke_test.py` -> `train.py` (60 iterations) -> checkpoint -> `coverage.py` ->
JSON + figure. All seven smoke checks pass, including preconditioning limits and the
sampler. Not yet run on a GPU and never trained to convergence, so no coverage number
exists yet.

Bug found by running it: `smoke_test.py` hard-coded arm `baseline3d`, which will not
exist on the cluster, so `train_diffusion.sh` would have died in its first minute. It
now reads `ARM` from the environment, and the SLURM script exports it.

## Status (original)

Written against the dataset, **not yet executed on a GPU** — this machine has no
torch. All nine files pass `py_compile`. Verified by running: `norm_stats.py`,
`data.py` (round-trips to physical units), `manifold.py` (99.97% in 2 components),
`joint_coverage` (calibrated / overconfident / underconfident all behave), and
`stitch.py` (exact reconstruction to 1.2e-7, seam metrics above). The UNet
forward/backward, the sampler and the training loop are unexecuted — expect shape
bugs on the first run; `smoke_test.py` catches them.

Not built: the image-domain stage, and a WGAN-ViT baseline for comparison.

# Baseline — Morovati et al. 2025 (residual WGAN-GP)

Reimplementation of the reference method, to compare against the diffusion arm:
[doi 10.1088/1361-6560/adaf71](https://iopscience.iop.org/article/10.1088/1361-6560/adaf71),
"Patch-based dual-domain photon-counting CT data correction with residual-based WGAN-ViT".

```
models.py        Generator (residual conv), Discriminator (WGAN critic), SmallViT
losses.py        WGAN-GP critic loss + generator loss (adversarial/MSE/RMAE/perceptual)
image_domain.py  second domain: TV-L1 (Chambolle-Pock) then guided filtering
metrics.py       PSNR / SSIM / RMSE, pure numpy
train.py         training loop
evaluate.py      their metrics, plus the manifold residual
```

```bash
python3 train.py --arm baseline3d --iters 200000 --batch 64 --lam_perc 0
python3 evaluate.py --ckpt ../../outputs/wgan_baseline3d_Y/ckpt.pt \
                    --arms baseline3d baseline3d_pu
```

Shares `diffusion/data.py` and the frozen normalisation from `diffusion/norm_stats.py`,
and `evaluate.py` uses the same `fixed_eval_set` patches as `diffusion/coverage.py`,
so the two arms see identical data.

---

## What the paper actually specifies

Checked against the text, because two of these are commonly misreported:

- **The generator is fully convolutional** — residual blocks, skip connections,
  transposed convolutions. **Not a ViT.** The ViT appears only as the perceptual-loss
  feature extractor, despite "WGAN-ViT" in the title.
- **"Residual-based"** = "the final output produced by adding the processed data back
  to the input", so `out = in + f(in)`.
- **WGAN-GP**, gradient penalty, not weight clipping
  ([Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028)).
- Discriminator: 4 residual blocks, 64 -> 128 -> 256 -> 512, then dense.
- Loss: adversarial + MSE + relative MAE + perceptual. Reconstruction terms ~1000,
  perceptual 10, eps 1e-4.
- Adam, lr 1e-4, 40 epochs. Batch size not stated.
- **The generator is deterministic** — one output per input.
- **No uncertainty is estimated anywhere in the paper.** PSNR / SSIM / RMSE only.

## Two places this cannot be exact

**The perceptual loss.** They train the ViT *from scratch on PCCT data*, explicitly
rejecting ImageNet pretraining: "Pre-trained CNNs are typically trained on ImageNet
dataset, which may differ significantly from our PCCT dataset." So there is no
checkpoint to download. `SmallViT` is provided but **must be trained** before
`--lam_perc 10` means anything — with random weights the term is pure noise, and
`train.py` prints a warning. Default is `--lam_perc 0`; say so in the write-up.

**Loss-weight assignment.** The paper's mapping of symbols to terms was ambiguous in
the text we could access (adversarial vs MSE both reported at ~1000). All four weights
are flags rather than hard-coded, so the ablation is cheap if it matters.

**The RMAE denominator.** The paper's relative MAE is `|G - p| / (p + eps)`, eps = 1e-4,
applied to projection data, which is positive. Our networks work in standardised log1p
space, where values cross zero, so the literal formula concentrates the loss on
near-zero pixels — measured, the top 1% of pixels carried **34%** of the RMAE weight and
RMAE outweighed MSE ~5x. `losses.py` therefore computes RMAE on **physical counts** with
the denominator floored at 1 count (`--rmae_floor`). That is a deliberate deviation:
state it, don't hide it. Found only because the loss was actually run — it compiles
either way.

## Image domain

`image_domain.py` implements the second stage: TV-L1 by Chambolle-Pock, then guided
filtering ([He et al.](https://kaiminghe.github.io/eccv10/)) using a virtual
"integrating" bin — the sum over all energy channels — as the guide. Using the summed
bin is the point: it carries the same anatomy at much better SNR, so edges come from it
while each energy bin keeps its own contrast.

Verified on a piecewise-constant phantom (3 bins, sigma = 0.25 noise):

| | RMSE | edge step (truth 0.50) |
|---|---|---|
| noisy input | 0.2494 | 0.5911 |
| TV-L1, lam=0.6 | 0.0528 | 0.3535 |
| TV-L1 + guided filter | **0.0509** | 0.3410 |

TV softens the edge somewhat (0.35 vs 0.50) — a real trade-off of the method, not a bug.

## What this baseline is for

It is the **point-estimate comparison**. Expect it to beat the diffusion posterior *mean*
on RMSE — it is trained with a lambda=1000 MSE anchor, which is exactly the objective
RMSE measures. That is not a defeat; a calibrated posterior and a minimum-RMSE estimate
are different objectives. Report both, and state it before a reviewer raises it.

What it cannot do is produce a coverage number, because the generator is deterministic.
`evaluate.py` prints `coverage: N/A` rather than inventing one. It does report the
manifold residual, which both methods can be scored on.

## Status (updated 2026-09-14)

**Executed end to end on CPU** with torch 2.14 in an isolated venv, on a tiny 4-phantom
clean arm: `train.py` (30 iterations) -> checkpoint -> `evaluate.py` -> JSON. Shapes,
the WGAN-GP gradient penalty, checkpoint save/load and the metrics all run. Not yet
run on a GPU, and never trained to convergence, so no number it has produced means
anything yet.

## Status (original)

**Not executed** — no torch on this machine. All six files pass `py_compile`.
Verified by running: `image_domain.py` (table above; a sign error in the Chambolle-Pock
primal step, `u - tau*div` instead of `u + tau*div`, made TV diverge to RMSE 92 — fixed
and retested) and `metrics.py` (identical inputs -> SSIM 1.0000; degrades correctly with
noise). The networks, losses and training loop are unrun; expect shape bugs on the first
GPU run.

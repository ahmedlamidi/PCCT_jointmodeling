# Progress report — 14 September 2026

Project: calibrated uncertainty under forward-model mismatch in photon-counting CT
(IPMI 2027). Today's step is the baseline comparison on one arm: a conditional
diffusion model (EDM) against a reproduction of the WGAN of
[Morovati et al. 2025](https://doi.org/10.1088/1361-6560/adaf71).

Arm `baseline3d_pu_matched`: PcTK 3.2 CdTe detector, 30 ns paralyzable pile-up,
noise-free labels, 9 closed energy bins (20–29 … 100–109 keV), 14 / 3 / 3 train /
val / test phantoms, 180 parallel-beam views. Trained on HiPerGator, one NVIDIA L4 GPU
per model.

## Status

| Item | State |
|---|---|
| Data generation | Done |
| Diffusion model | Trained, 200,000 iterations (job 42078987); evaluated on the test split |
| WGAN baseline | Training: 46,000 of 1,086,750 iterations (4.2%), about 38 h left at 7.7 it/s |
| Windowed SSIM and FBP-slice evaluation | Script ready; not yet run on the cluster |
| Diffusion vs WGAN table | Waits for the WGAN |

## Training curves

All numbers below are read from the SLURM logs in `logs/`. Each point in the figures
is the mean over 200 iterations; the line is a rolling median, for the eye only.
One exception: the logs print the WGAN critic loss for the **last** critic step before
each log line, not a 200-iteration mean. That curve is single steps, so it is noisier;
read its trend from the rolling median.

| Figure | Shows |
|---|---|
| [training_curves_overview.png](training_curves_overview.png) | Every curve on one page |
| [figures/01_diffusion_loss.png](figures/01_diffusion_loss.png) | Diffusion denoising loss |
| [figures/02_diffusion_speed.png](figures/02_diffusion_speed.png) | Diffusion training speed |
| [figures/03_wgan_mse.png](figures/03_wgan_mse.png) | WGAN MSE term |
| [figures/04_wgan_rmae.png](figures/04_wgan_rmae.png) | WGAN relative-MAE term |
| [figures/05_wgan_generator_total.png](figures/05_wgan_generator_total.png) | WGAN generator loss, weighted total |
| [figures/06_wgan_adversarial.png](figures/06_wgan_adversarial.png) | WGAN adversarial term |
| [figures/07_wgan_critic.png](figures/07_wgan_critic.png) | WGAN critic loss |
| [figures/08_wgan_speed.png](figures/08_wgan_speed.png) | WGAN training speed |
| [figures/training_summary.txt](figures/training_summary.txt) | Iterations, jobs, speed, time left |
| [figures/09_unet_architecture.png](figures/09_unet_architecture.png) | Diffusion UNet: levels, down/upsampling, skip connections (13.76 M parameters) |

The diffusion loss and the WGAN terms are different objectives in different units.
They must not be compared by value.

**Diffusion.** The loss fell from 0.070 (mean of the first 10k iterations) to 0.0070
(10k–20k), then slowly: 0.0039 at 90k–100k, 0.0035 at 190k–200k. The last 100k
iterations gained about 10%. The loss cannot reach zero: it averages the denoising
error over random noise levels, and at high noise the clean patch is not
recoverable ([Karras et al. 2022](https://arxiv.org/abs/2206.00364)). So a flat
training loss does not measure sample quality; the evaluation below does.

**WGAN.** Not converged. The MSE term is still falling about 25% per 10k iterations
(0.00099 at 20k, 0.00073 at 30k, 0.00059 at 40k, 0.00054 at 46k). The critic loss
rose from about −9 to about −0.7 (single-step values; the rolling median shows the
same trend). In WGAN-GP the negative critic loss approximates the
Wasserstein distance between real and generated data, so this means the critic
separates them less and less ([Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028)).
The adversarial term has been flat at about −1.6 since roughly 5k iterations.

**Speed.** A data-loader fix today (all training phantoms held in memory) raised the
WGAN from 2.8 to 7.7 iterations per second (medians of the two jobs). The first job
(42078988) was cancelled at 25,000 for this; the second (42125890) resumed from the
20,000 checkpoint. The figures use the second job where they overlap.

## Diffusion model on the test split

From the end of `logs/edm_42078987.out`: 256 fixed test patches drawn across all three
test phantoms, 256 posterior samples each.

| Nominal coverage | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| Per-bin (mean of 9 bins) | 0.438 | 0.530 | 0.619 | 0.708 | 0.807 | 0.870 | 0.941 |
| Joint (all 9 bins, on the 2-D manifold) | 0.517 | 0.610 | 0.720 | 0.834 | 0.915 | 0.952 | 0.985 |

With 256 samples a perfectly calibrated model reads slightly below nominal, because
the intervals come from sample quantiles:

- about 0.893 per bin at the 0.90 level. My own calculation: a sample-quantile interval
  covers L × 255/257 on average. It matched a simulation at 64 samples.
- about 0.886 jointly, from the finite-sample table in `src/diffusion/manifold.py`.

Against those references, the per-bin intervals are **too narrow (overconfident)** and
the joint regions are **slightly too wide**.

- **Manifold residual:** samples 0.689 against 0.0129 for the truth (53×). The ideal
  9-bin counts depend on only two numbers, cm of soft tissue and cm of bone, so a
  physical count vector lies on a 2-D surface. The samples sit far off it.
- **Posterior mean:** RMSE 4.26 counts, PSNR 54.1 dB, SSIM 0.9999. That SSIM is a
  single global window over all patches and bins. It reads about 1 for any sensible
  prediction and is being replaced (next section).

A likely contributor to the large residual, **not yet tested**: the 20–29 keV bin
is nearly empty through the head. This repo's forward model gives an ideal 0.28–0.85
counts behind 15–17.5 cm of soft tissue, and 0.04–0.06 with bone on the path. The
residual is measured on log counts, so near-zero values there can dominate it. The
check is the residual per bin, with median and percentiles rather than the mean.

Morovati et al. report no uncertainty, only PSNR, SSIM and RMSE
([Morovati et al. 2025](https://doi.org/10.1088/1361-6560/adaf71)). The coverage and
residual rows are the part only the diffusion model can fill.

## Other changes today

- **Windowed SSIM** per energy bin, as in [Wang et al. 2004](https://doi.org/10.1109/TIP.2003.819861).
  Matches `skimage` exactly. Computed on the test patches and on FBP-reconstructed slices.
- **FBP with a Hann-apodised ramp.** The plain ramp turned noisy count sinograms into noise
  (correlation with the true image 0.03–0.17; 0.74–0.96 with the Hann window; measured
  on an older local arm). Standard remedy:
  [Kak & Slaney, ch. 3](https://www.slaney.org/pct/pct-toc.html).
- **The 20–29 keV bin is excluded from the slice SSIM.** Its line integral is not
  measurable at this flux (counts above). This is physics, not a bug.
- **Sinogram figures.** A time series of one detector channel over all 180 views, with
  input, clean label, model output and ±2 std.
- **Training CSV logs** (`train_log.csv`, resume-safe) for all future runs.

## Next

1. Let the WGAN finish (one or two resubmits after TIMEOUT), then run the comparison table.
2. Run the windowed-SSIM evaluation for both models (`slurm/hipergator/ssim_diffusion.sh`).
3. Break the manifold residual down per bin, with median and percentiles.
4. Generate the mismatch arm (pile-up off) and evaluate both trained models on it.

## Not yet executed

- The windowed-SSIM and FBP-slice evaluation has run only with stand-in models on this
  machine. The model-loading code has not run anywhere yet.
- The training CSV logging is written and unit-tested, but has not run in a training job.

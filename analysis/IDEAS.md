# Idea list

Design ideas raised for the IPMI 2027 paper, each with the evidence we actually
measured and the prior art. Status is one of: **DONE**, **DO THIS**, **TEST FIRST**,
**REJECTED**.

---

## 1. Add a time axis to the condition — 16 x 16 x 9 x 5

**REJECTED on physics, weak denoising case.**

Idea: stack 5 adjacent views so the network can "see across time" for pile-up.

Why it does not do what it sounds like:

| | |
|---|---|
| dead time | 30 ns |
| pulse length | 25 ns |
| one view (frame) | 250 us = 250,000 ns |
| frame / dead time | **8333x** |
| P(a pulse straddles a frame boundary) | 1e-4 |

The detector recovers 8333x over between views, so pile-up state does not carry
from view v to view v+1. The nanosecond structure is integrated away at readout,
before the data exists. No architecture recovers it.

What a 5-view stack *would* give: adjacent views are near-repeat measurements
(path changes 0.3% per view at 180 views, 0.9% over 5), so ~5 independent noise
realisations of the same rate, i.e. ~sqrt(5) better rate estimate. That is
denoising, and it overlaps with what spatial context already buys.

Prior art *does* use temporal input for pile-up — but at the **pulse waveform**
timescale (nanosecond samples), not across views:
- Huang, Zheng, Zhu, Trigano, Bykhovsky & Chen, "Deep Learning Based Pile-Up
  Correction Algorithm for Spectrometric Data Under High-Count-Rate Measurements",
  *Sensors* 25(5):1464, 2025, [doi 10.3390/s25051464](https://doi.org/10.3390/s25051464)
  ([full text via PMC11902627](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902627/) —
  MDPI itself returns 403 to automated fetching).
  **Input:** Energy-Duration matrix — energy (1024 bins, to ~1666 keV) x pulse
  duration in time samples, from zero-crossing segmentation of the raw signal,
  with count rate embedded. **Output:** a corrected energy-spectrum histogram.
  **Key idea:** duration is the pile-up discriminator — merged pulses last longer,
  so an anomalous duration for a given energy flags a pile-up event.
  **Does not transfer:** this is gamma-ray spectroscopy (one detector, one spectrum
  out, no spatial dimension — the "2D" is energy x duration, not image axes), and
  we have no per-event duration. Our electronics discard it before readout.
- U-Net / LSTM / CNN comparisons on digitised waveforms to deconvolve
  superimposed pulses.

If your hardware can expose sub-frame time bins, this becomes a real option and
the papers above are the template. With 250 us frames it is not.

## 2. Condition on the estimated count rate — 1 extra channel

**TEST FIRST.** The cheap version of idea 1, and it has precedent.

Pile-up severity is a function of the rate. Hand the network the rate directly
instead of 4 extra copies of the patch. The *Sensors* 2025 paper above embeds
count-rate information for exactly this reason (again, unverified).

Caveat from our own data (idea 8): the rate is already recoverable from the 9
bins, so this may add nothing. Test with the ridge harness before building it.

## 3. Overlap and blend when stitching patches

**DONE** — [`src/diffusion/stitch.py`](src/diffusion/stitch.py).

Patches are already 16x16 at stride 8 (50% overlap, 4 patches per interior
pixel). Overlap alone is not enough; the patches have to be *combined*.

Measured, with each patch given an independent bias:

| | MAE | RMS | seam excess |
|---|---|---|---|
| plain average | 0.0989 | 0.1300 | **+0.0998** |
| Hann taper | 0.1020 | 0.1283 | **-0.0181** |

The taper does not improve average accuracy — it removes the *structure*. Plain
averaging leaves a gradient spike at every patch boundary; the eye locks onto
regular lines instantly. Report RMSE alone and you would wrongly conclude
blending does not matter.

Standard practice: overlap-tile, as in [U-Net (Ronneberger et al. 2015)](https://arxiv.org/abs/1505.04597).

## 4. Generate materials (V) rather than corrected counts (Y)

**DO THIS** (current default in `train.py`).

A credible interval is only meaningful on a physical quantity. Also sidesteps
the CRLB result: per-pixel decomposition is already at the bound (efficiency
0.95-0.96 below ~3 cm bone), so there is no headroom in the *point estimate* —
which is an argument for predicting a distribution instead.

## 5. Regenerate labels NOISE-FREE

**DO THIS, BEFORE ANY COVERAGE NUMBER.** Resolved 2026-09-09 by reading the
reference paper's methods (see idea 11).

Morovati's label is **noise-free**: "Poisson noise was added on a pixel-by-pixel
basis" to the *distorted* data only. Ours is a noisy ideal realisation that
additionally shares its seed with the input.

Evidence of the damage: the receptive-field sweep gained 29% from spatial
context on noisy input but **0.1%** on noise-free input — part of the "context"
gain was the model predicting the label's own noise. For a generative posterior
this is worse than for a regressor: it narrows the conditional distribution and
makes calibration look better than it is, which is the exact quantity we report.

Fix is one flag — `Forward.sample` already returns `mu_i`, the generator was
discarding it:

    python3 generate_baseline_dataset3d.py --label clean --closed_top_bin ...

## 11. Corrections after reading Morovati et al. methods (2026-09-09)

Obtained via IOPscience ([doi 10.1088/1361-6560/adaf71](https://iopscience.iop.org/article/10.1088/1361-6560/adaf71));
PMC serves a captcha to automated fetching. Four documented assumptions were wrong.

| | we assumed | they actually do | status |
|---|---|---|---|
| detector model | unmatched, unknown | **PcTK 3.2**, CdTe — same as ours | now MATCHED |
| label | noisy ideal, shared seed | **noise-free** ideal projection | **fix: `--label clean`** |
| geometry | fan beam (R=600, Rd=1080) | **parallel beam** ("for convenience and generality"); the real-data arm is cone beam | mismatch, ours is harder |
| top energy bin | 100-inf, open | thresholds to **110 keV**, nine CLOSED bins 20-29 ... 100-109 | **fix: `--closed_top_bin`** |
| spectrum | PcTK's shipped 120 kVp + 2 mm Al | **SpekCalc**, spans 12-120 keV; kVp not stated | minor |

The top-bin difference matters most for pile-up. Our open bin 9 collects the sum
events that drive the +943% air signal; their 110 keV cut throws those out of
every bin. Any pile-up comparison against their numbers must close the top bin.

Confirmed matched: 16 x 16 x 9 patches, 1,841,400 patches, 10 train + 5 test
phantoms, 180 projections.

## 6. r0 sweep — the one unrun validation

**TEST FIRST.** Regenerate `nCovE` at a different charge-cloud radius (~70 min
per point via `run_pctk('gennc')`) and check the sharing fraction scales as the
Gaussian geometry predicts.

Every charge-sharing check so far inspects a *fixed* table. This is the only one
that varies the model's own input parameter. We already recovered sigma ~ 16 um
from the edge:corner ratio (29-32) against the stated r0 = 24 um; a sweep would
confirm the model responds correctly to the knob rather than merely looking
plausible at one setting.

## 12. Baselines to implement

Two, both cheap, both needed:

1. **Morovati's generator** — deterministic conv net (residual blocks, skip connections),
   loss = adversarial + MSE + relative MAE + perceptual. The point-estimate comparison.
   Their discriminator/perceptual/ViT exist to undo MSE blur; under diffusion they are
   unnecessary, so this is a *baseline*, not a component to port.
2. **The quadratic.** From idea 8: a quadratic on one pixel's 9 bins hits R^2 = 0.99999
   on the noise-free map. If a network cannot beat that on noisy data, say so plainly.

Expect our posterior MEAN to lose on RMSE to a method trained with a lambda=1000 MSE
anchor. That is not a failure — different objective — but it must be stated before a
reviewer treats it as one.

## 7. Model-based baseline

**DO THIS.** [Taguchi et al., *Med Phys* 2022, "Model-based pulse pileup and charge
sharing compensation for photon counting detectors"](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15779)
— PcTK's own author doing joint pile-up + charge-sharing compensation. Closest
prior art to this project and the obvious non-learned baseline. A learned method
that does not beat it needs a different justification.

## 8. Reframe the paper: the distortion is easy, the uncertainty is hard

**DO THIS.** The strongest finding we have.

From the 9 energy bins at a **single pixel**, noise-free, one view:

| task | model | R2 |
|---|---|---|
| undo pile-up + sharing | quadratic, 9 bins | **0.99999** |
| recover (a, b) material paths | quadratic, 9 bins | **0.99999** (0.019 cm soft, 0.0089 cm bone) |

0.19 mm on a 21.5 cm path, with no neighbours and no time axis. Consistent with
the receptive-field sweep (0.1% gain from context on clean data) and the CRLB
sweep (efficiency 0.95-0.96).

So a network that learns the mean correction is solving the easy half. The paper
should rest on: **calibrated uncertainty, and how it degrades under
forward-model mismatch.**

CAVEAT: that test used noise-free means from ONE view, so the rays trace a 1-D
curve through the 2-D (a,b) plane. Rerun on the full noisy `baseline3d_pu`
before relying on it (~1 h, reuses the ridge harness).

## 9. Energy-domain stamping so rho and tau compose

**Not built.** `rho` stamps binned counts; pile-up acts on the deposited-energy
spectrum. Doing both rigorously means stamping in the energy domain
(191 bins x 9 pixels) and thresholding after pile-up. Until then, sweep one
mismatch axis at a time.

## 10. Pin the frame exposure time

**Blocked on external information.** Currently assumed 4000 views/s (250 us).
lambda*tau scales linearly with it — air spans 0.6 to 4.9 over a plausible
range, which is the difference between pile-up being a minor correction and the
dominant distortion. Any pile-up result must state the assumption.

Related: pile-up turns out to be an **air/low-attenuation** effect at our
operating point — through the head it is only 2-7% on top of charge sharing,
versus -83% to +943% in air. If that holds, the pile-up mismatch arm may be a
weak test, since the patient-bearing rays barely move. Check at 125 us.

# The pipeline, stage by stage, and every decision in it

Companion to `README.md`. This is the "why", not the "how to run".
Last updated after the pile-up validation and the 3D dataset build.

---

## Stage 0 — What PcTK actually gives us

One 4 GB array, `m3_nCov3x3E`, of shape **(1719, 1719, 170)**.

- `170` — incident photon energy, 1–170 keV (`v_E1`)
- `1719` — **9 pixels × 191 deposited-energy bins**, energy fastest:
  `index = 191*p + eo`, with `p = ix + 3*iy`, so `p=4` is the centre pixel
- `v_Eo` runs 0–190 keV

For each incident energy it is a **covariance matrix** over "counts at
(pixel, deposited energy)". Its **diagonal is the mean response**; the full
matrix carries the noise correlations.

**Decision: cache only the diagonal.** `cache/diagE.npy` is (170, 1719), 2.3 MB
against 4 GB, extracted once in ~7 minutes of I/O. Every mean-level quantity
needs only the diagonal. The full matrix is re-read only when covariance at new
thresholds is required (`rebin_cov.py`).

**Verification:** binning the diagonal into PcTK's own 20/50/65/80 keV windows
reproduces the shipped `nCov3x3w` to **3.3e-16**. This is the anchor for
everything downstream.

### The q-branch decomposition

`nCovE` ships `m2_SRE_q0/q1/q2` and `v_Pr_pq`, splitting the response by
fluorescence branch:

    diag(nCovE(:,:,E)) = Pr(E,1)*q0(:,E) + Pr(E,2)*q1(:,E) + Pr(E,3)*q2(:,E)

Identified empirically:

- **q0** — photoelectric, no fluorescence. `Pr(:,1)` is exactly 1 below 26.7 keV
  (Cd K-edge), 0.352 above it, 0.277 above the Te K-edge (31.8 keV).
- **q1** — **K-escape.** At E₁=100 keV it peaks at 75–76 keV (= 100 − 24.5) and
  is *exactly zero* at the photopeak.
- **q2** — fluorescence reabsorbed nearby; photopeak plus a tail.
- `sum(v_Pr_pq, 2)` is the **quantum efficiency**: 0.997 at 30 keV, 0.785 at 100,
  0.616 at 120. The deficit is transmission.

**Two fixes the manual omits:**

1. **Quadrant symmetry.** Only pixel indices 4, 5, 9 (1-based) hold valid data;
   fill 1,2,3,6,7,8 by symmetry. Without this, per-pixel sums come out 0.37
   where truth is 0.0091.
2. **An overflow accumulator above ~175 keV.** The q-arrays park normalisation
   mass there; the covariance diagonal is zero there. Zeroing `v_Eo > 175` takes
   the identity from 1.6e-1 to **1.3e-4**. It matters because those bins fall in
   the top window.

This is what makes `FLUOR` a free toggle — K-escape switches off with no MATLAB.

---

## Stage 1 — Phantom → line integrals

`phantom.py` builds **fraction maps** (0–1), not densities. `head()`/`variants()`
are 2D; `head3d()`/`variants3d()` stack axial slices as solids of revolution.

`projector.py` turns them into line integrals in **cm**.

**Decision: project in parallel geometry, then resample to fan.** Direct fan
ray-tracing over 1854 × 2000 rays is 2.6 G samples, too slow in numpy. Rotate
and sum columns instead (fast, exact up to interpolation), then resample with

    fan ray (beta, gamma)  ->  parallel ray (theta = beta + gamma, t = R sin gamma)

**Geometry** (from PcTK `v_N0.mat`): R = 600 mm source-to-isocentre,
Rd = 1080 mm source-to-detector, 1854 channels at 225 µm equiangular.
Magnification 1.8, so a channel covers 125 µm at isocentre and the detector
spans ~23 cm there — a head fills nearly all of it.

**Verified against analytic geometry**, not just self-consistency:

| | predicted from the ellipse | measured |
|---|---|---|
| max soft path | 2 × 0.87 × 125 mm = 21.75 cm | 21.63 cm |
| max bone path (skull chord) | 2√(106.25²−98.75²) = 7.84 cm | 7.97 cm |

**The FBP constant is wrong by ~1.85** (suspiciously close to the magnification
1080/600 = 1.8). `roundtrip_check()` returns it. Shape error is 1.51% at 360
views, which is normal for linear-interpolation FBP. Only affects display —
datasets are built from `forward`, never from `fbp`.

### Two phantom bugs, both fixed

Shepp-Logan ellipse **values add where they overlap**. Two consequences:

1. Bone exceeded 1.0 (pure cortical bone) in overlap regions → now clipped.
2. Worse: `variants()` jittered the skull's outer and inner ellipses
   **independently**, so the outer could grow while the inner shrank. That made
   the wall several times too thick and gave tangential rays through
   **13–16 cm of bone** — more than a skull is wide, and outside the pile-up
   grid's range. Now the outer ellipse and a **wall thickness** are drawn, and
   the inner is derived. Worst bone path across 15 variants went 15.93 → **8.98 cm**,
   and rays over 8 cm from 2.8% → 0.07%.

---

## Stage 2 — Line integrals → attenuated spectrum

    S_att(E) = S0(E) * exp(-mu_tissue(E)*a - mu_bone(E)*b)

**This is the load-bearing observation.** For a two-material phantom, *every*
downstream quantity — counts, covariance, pile-up, decomposition noise — is a
function of just **(a, b)**. Two numbers per ray.

Consequences used throughout:

- The mean distortion is a **15-term quadratic** in log-counts, to 0.05%.
- Pile-up is precomputed on an (a, b) grid and interpolated.
- The spatial-noise matrix square root is gridded the same way.

---

## Stage 3 — Spectrum → deposited energy (PcTK)

    R[E1, Eo] = sum over the 9 pixels of diag(nCovE)

**Decision: sum over the 3×3 neighbourhood.** Under uniform illumination,
shift-invariance means what a pixel receives from its neighbours equals what it
scatters to them, so the 9-pixel sum is the correct counts-per-pixel. This is
why totals exceed 1 count per incident photon.

`PIXELS='center'` keeps only pixel 5. That is a **diagnostic**, not a detector
configuration: it discards real counts (51.7 → 39.4 behind bone).

---

## Stage 4 — Pile-up (rate-dependent) — **validated**

PcTK is rate-**independent**; a grep of its source finds no dead time, flux or
count-rate parameter anywhere. Pile-up is a separate stage cascaded after it,
per Cammin/Taguchi ordering — physical, since charge collection happens in the
sensor and pile-up in the readout.

**Decision: pile-up acts on the deposited-energy spectrum, never on window
counts.** Thresholding destroys exactly the information pile-up needs, so the
cascade uses `nCovE` (191 bins), not `nCovw`.

Three implementations:

- **`pileup.py`** — first Monte Carlo. Square pulse window, energies summed.
  Superseded as a reference but kept.
- **`pileup_mc_ref.py`** — the **reference**. Yang's Algorithm 1: discretise
  time, build the delta train, convolve with a triangular pulse kernel, run a
  non-paralyzable digital counter. Retriggering falls out naturally.
- **`pileup_yang.py`** — analytical (Yang, Pelc & Wang, Med Phys 2025,
  doi 10.1002/mp.17746). 0.4 s per point.

### Reconstructed equations

The available copy of the paper rendered its equations as images, so Eq 16–20
were not readable. Taken **directly from prose**: `P(m) = Poisson(λτ)`, the
triangular-pulse approximation, "output = peak of the summed signal", and the
compound-multinomial binning
`mu = E[N]*p`, `C = E[N]*(diag(p) − p pᵀ) + Var[N]* p pᵀ`.

**Reconstructed by algebra following their description:**

- `tau_snp(E) = t_p + (T − t_p)*(1 − E_t/E)` ("peak, then decay to threshold")
- free time is 0 w.p. `P0`, else `Exp(λ)`, giving `E[R] = tau + (1−P0)/λ` and
  `Var[R] = (1−P0)(1+P0)/λ²`
- renewal asymptotics `E[N] = T/E[R]`, `Var[N] = T·Var[R]/E[R]³`
- retrigger probability from the order statistic `1 − ((tau − tau_snp)/tau)^m`

### What the earlier disagreement was

Two things, **neither a flaw in the physics**:

1. **Mismatched pulse shapes** — the old MC used a square window, the analytical
   a triangle. Matching them closed most of the gap.
2. **A degenerate parameter choice.** With `tau_dead` = 20 ns and pulse length
   `T` = 25 ns, a pulse is still at ⅓ of its peak when its own dead time
   expires, so **every photon retriggers itself** and bin 1 inflates ~38%.
   `tau >= T` makes it vanish:

       tau=20 ns -> tail at 0.33 of peak   SELF-RETRIGGERS
       tau=25 ns -> tail at 0.00           clean

   **`T_ns` now defaults to 30** so this cannot be hit by accident.

### Agreement, with valid parameters (t_p=10, T=25, tau=30 ns)

| λτ | max error | Fano MC / analytical |
|---|---|---|
| 0.009 | 2.2% | 0.99 / 1.00 |
| 0.095 | 5.7% | 0.96 / 0.99 |
| 0.229 | 9.6% | 0.98 / 0.97 |
| 0.567 | 16.0% | 0.93 / 0.92 |
| 2.448 | 36.6% | 0.78 / 0.78 |

**Count statistics agree everywhere** — that is the part coverage work needs,
and the part Yang adds over Taguchi 2010. The *spectrum* drifts with rate. Yang
report the same failure mode themselves. This reconstruction is worse than their
published numbers (18% at λτ=1.15 vs their ~11%).

**Decision: `backend='mc'` is the default.** Since the pile-up operator is
interpolated from a grid anyway, and the grid is a one-time cost, build it with
the Monte Carlo and rely on none of the reconstructed equations. `'yang'` remains
available for speed at low rate.

Counts become **sub-Poisson** under pile-up: Fano 0.63–0.95 in air, ≈1.0 behind
the patient. Generating Poisson noise where reality is sub-Poisson would make
every reported interval wrong before training starts.

---

## Stage 5 — Thresholding

`np.digitize(v_Eo, eth) - 1`. Deposited energy below the lowest threshold
returns −1 and is **uncounted** — physically right, and why totals are far below
the raw diagonal sum (which contains a large delta near 0 keV for neighbour
pixels meaning "no charge deposited here").

**Decision: bin in the energy domain, not by regenerating in MATLAB.** The
response is kept at 191 one-keV bins, so *any* threshold scheme is a rebinning.
`ETH` takes any number of windows.

The **covariance** does not rebin so cheaply — it needs the off-diagonal terms.
`rebin_cov.py` streams the 4 GB file and computes `C_w = Aᵀ C_E A` with an
indicator `A`, cached per threshold set (~10 min each).

**Validated.** `--validate` rebins at PcTK's own thresholds and compares against
the shipped table: **max error 4.857e-17** against a scale of 0.829 — less than
one double-precision step. This also confirms the inferred index layout
(`191*p + eo`, energy fastest) is right.

The 9-bin build (20–109 keV) passes four further checks:

| check | result |
|---|---|
| symmetry | 6.94e-18 |
| PSD (min eigenvalue) | −3.4e-16 vs scale 0.73 |
| diagonal vs independently cached `diagE` | 1.67e-16 |
| total counts vs the 4-bin binning | 3.33e-16 |

The third is a genuinely independent cross-check: two separate code paths to the
same numbers. Bins above the incident energy are exactly zero, the photopeak
lands in the correct bin at every energy, and the per-energy sums match the
4-bin totals computed before any of this existed.

---

## Stage 6 — Noise, and the paired branches

```python
z = rng.standard_normal((npix, Nl))          # ONE draw
ideal     = mu_ideal + sqrt(mu_ideal) * z    # Poisson approximation, diagonal
distorted = mu_dist  + L @ z                 # L = matrix square root of the covariance
```

**Decision: share `z` between branches.** A pair then differs *only* by detector
physics, never by dice.

**Decision: `mean = diag(covariance)`.** Not an approximation — it is how PcTK
defines the model, and `script_workflow_PcTK.m` does the same.

**Decision: eigendecomposition, not Cholesky.** The covariance is occasionally
not quite PSD; PcTK's own code calls `sup_offdiag` to repair it. `eigh` with
eigenvalues clipped at 0 never fails.

**Verification:** `(noisy − mean)/sqrt(mean)` over a full generated sinogram has
mean 0.0002, sd 0.9998.

### Spatial versus spectral correlation

`pipeline.py` collapses the neighbourhood covariance to one pixel by summing the
nine diagonal blocks. That keeps **spectral** correlation and discards
**spatial**. For patch-based networks this matters — spatially correlated noise
looks completely different at patch level.

`noise_spatial.py` keeps both, by stamping: draw the full 36-vector per source
pixel and add it into the 3×3 neighbourhood, accumulating overlaps.

**Index bug, fixed.** PcTK packs the vector as `bin + Nl*p` with `p = ix + 3*iy`,
so **bin varies fastest**. The original `reshape(Nch, Nrow, Nl, 3, 3)` put bin
*slowest*; the stamped mean came out `[2.57, 13.14, 7.59, 2.57]` against the
correct `[25.11, 10.99, 8.03, 7.59]` — the same numbers scrambled across bins.
C-order reshape must be **(iy, ix, bin)**. After the fix the stamped mean matches
the pipeline mean to **7.1e-15**.

**Single-row artifact, fixed.** `measured_correlation` used one detector row, so
each pixel received only 3 of its 9 neighbour contributions and both mean and
correlation were deflated. It now requires `nrow >= 3` and reads the middle row.

### The rho knob, calibrated

    C(rho) = rho*C + (1 - rho)*BlockDiag(C)

PSD for all rho in [0,1] (a convex combination of two PSD matrices).
**rho scales noise correlation only** — the diagonal, hence the mean, is
untouched. Real anti-coincidence also recombines charge and shifts the mean; the
mean-level analogue is `PIXELS='center'`.

Measured adjacent-channel correlation, against the analytic Gate 2 prediction:

| | air | 20 cm brain | 20 cm + 2 cm bone |
|---|---|---|---|
| analytic (Gate 2) | 0.0639 | 0.0911 | 0.1005 |
| rho = 1.00 | 0.0580 | 0.0878 | 0.0982 |
| rho = 0.50 | 0.0262 | 0.0424 | 0.0480 |
| rho = 0.25 | 0.0103 | 0.0197 | 0.0229 |
| rho = 0.00 | −0.0056 | −0.0030 | −0.0021 |

At rho=1 the simulation reproduces the analytic prediction to 91–98%; the
residual is about 1σ of sampling noise at 30,000 channels. rho scales the
correlation almost exactly linearly.

**Therefore a measured <0.01 on real hardware corresponds to rho ≈ 0.1** — the
endpoint of p(S), derived rather than assumed. The detector's anti-coincidence
removes roughly 90% of the spatial sharing correlation PcTK predicts.

**The two mismatch axes do not compose.** `rho` stamps *binned counts*; pile-up
acts on a pixel's deposited-*energy* spectrum. Doing both rigorously means
stamping in the energy domain (191 bins × 9 pixels) and thresholding after
pile-up. Not built. Sweep one axis at a time.

---

## The flux question

The shipped `v_N0.mat` holds **1,414,609 photons/pixel/view**; the software
note's own arithmetic gives ~11,900–14,300.

A full MATLAB run settled it: generated air counts were
`[1006437, 248121, 110108, 67305]`, matching the authors' own commented values
at `3_src/script_workflow_PcTK.m:304` — `[1006400, 248100, 110100, 67300]` — to
**0.004%**. **The shipped value is correct for reproducing the example**, and an
earlier claim here that it was "100× too high" was wrong.

What is true is narrower: read as photons/pixel/view at 4000 views/s it implies
~5.7e9 cps/pixel, which is unphysical. That matters **only for pile-up**.
`CFG['N0'] = 12000` is the physically consistent default for rate-dependent work.

Frame exposure time is still the blocking unknown for turning real measured
counts into a rate.

---

## Matching Morovati et al. 2025

`generate_baseline_dataset3d.py`. Their patch count gives a useful inference:
1,841,400 / (10 phantoms × 180 projections) = **exactly 1023 per projection**,
consistent with a ~256 × 272 detector plane at stride 8. So they patch the
**detector plane (u, v)**, not the sinogram, at stride 8. Both adopted.

**Matched:** 10 train / 5 test, 3D Shepp-Logan variants, 180 projections, 9 bins
(20–109 keV, 10 keV steps), 16×16×9 patches, (u,v) patching, stride 8, and
1,863,000 patches against their 1,841,400 (+1.2%).

**Not matched:** cone beam (multi-slice fan instead, one row per axial slice);
detector model and parameters (unknown — PcTK 225 µm / 1600 µm CdTe used);
spectrum (unknown — PcTK's 120 kVp; their 109 keV top bin implies 110–120 kVp);
flux and dead time (unknown); spatial sampling (our pixels are finer, so a
16×16 patch spans ~2 mm at isocentre against their likely ~8 mm).

A 256-channel detector is **not** available: PcTK's 225 µm pitch over 1854
channels spans 23 cm at isocentre and the head fills nearly all of it, so
cropping to 256 would capture 14% of the head width. Their detector must have
coarser pixels. Patch *count* was matched instead of detector width.

**The gap that could invalidate a comparison: how they defined the ground-truth
label.** Ours is "same spectrum, perfect thresholds, same QE, Poisson noise,
same seed". If theirs is a noise-free reference, a high-dose acquisition, or an
energy-integrating equivalent, the two tasks differ and no number is comparable.
That and the detector parameters are what to request from the authors.

### A chunking trap

`--chunk` originally counted **views**. In the 2D generator a view is 1854 rays;
in the 3D generator it is `nrow × nch` = 59,000 rays. `--chunk 60` therefore
requested 3.56M rays, and `Forward.sample` builds a `(170, nrays)` float64
spectrum — **4.8 GB** — which OOM'd instantly with no traceback. Chunking is now
on **rays** (`--maxrays`, default 300,000 → ~403 MB) and the peak size is printed
at startup.

---

## Verified versus assumed

| Claim | Status |
|---|---|
| E-domain binning reproduces shipped `nCov3x3w` | **3.3e-16** |
| `rebin_cov` at PcTK thresholds vs shipped table | **4.9e-17** |
| 9-bin covariance diagonal vs cached `diagE` | **1.7e-16** |
| Generated air counts vs authors' documented values | **0.004%** |
| q-branch identity after symmetry + overflow fixes | **1.3e-4** |
| Noise z-score over a full sinogram | mean 0.0002, sd 0.9998 |
| Stamped mean vs pipeline mean | **7.1e-15** |
| Spatial correlation at rho=1 vs analytic Gate 2 | 91–98% |
| Forward projector vs analytic ellipse geometry | ~1% |
| Yang analytical pile-up | 2% at λτ=0.01, **37% at 2.4** |
| p-code runs on MATLAB R2025b | confirmed |
| FBP absolute µ scaling | **wrong by ~1.85** |
| Yang Eq 16–20 | **reconstructed from prose** |

## Measured results

- **Noise penalty** from cross-talk: flat **2.18–2.40×** on decomposed bone.
- **Estimator efficiency**: 0.95–0.96 of the CRLB below ~3 cm bone — essentially
  optimal, no headroom. Above that, "efficiency > 1" is *severe bias*
  (+1.6 cm on a 5 cm truth), not an estimator beating the bound.
- **PcTK predicts 6–12× more neighbour correlation than real hardware shows**
  (0.064–0.100 against a measured <0.01).
- **Cross-talk is a bottom-window phenomenon**: behind 20 cm tissue + 2 cm bone,
  52% of window-1 counts land outside the centre pixel; windows 2–4 are 98–100%
  centred.
- **Cascade at a central ray** (4 bins): ideal `[3.0, 15.4, 15.5, 20.4]` →
  +sharing `[23.4, 16.5, 12.4, 12.6]` → +K-escape `[34.5, 14.9, 10.5, 9.6]` →
  +pile-up `[36.4, 15.7, 10.8, 10.0]`. Sharing is about two-thirds of the
  bottom-bin contamination, fluorescence one-third, and they compound.

## Known gaps

1. `rho` and `tau` do not compose (needs energy-domain stamping).
2. `rho` moves noise only, not the mean.
3. `demo_end2end.py` on disk is still the version that OOMs.
4. FBP absolute µ scaling wrong by ~1.85 (display only).
5. Only one operator point for `r_0` / `dz`; a grid needs ~70 min of MATLAB per
   point, and is blocked on sensor thickness anyway.
6. No models — no WGAN-ViT, no CycN-Net, no flow matching, no torch code.
7. No coverage experiment (P4).
8. Blocked on external information: frame exposure time (blocking), CdTe sensor
   thickness, chicken-leg dataset, further walnut records.

# Figure index

Names are prefixed by what the figure answers:

| prefix | question |
|---|---|
| `physics_` | what does the detector do to a photon? |
| `cascade_` | what does each effect add, in order? |
| `signal_` | how many counts, per bin, per stage? |
| `data_` | what does the generated dataset look like? |
| `recon_` | what does it do to an image? |
| `stats_` | quantitative analysis |

Run everything from `analysis/src/`.

---

## physics — detector behaviour

| file | shows | script |
|---|---|---|
| **`physics_charge_sharing_evidence.png`** | **the meeting figure.** One 100 keV photon lands on one pixel; 18% of its energy is recorded in pixels it never hit. Plus the smeared spectrum, and sharing vs a geometry-only prediction across the spectrum | `fig_sharing_evidence.py --E 100` |
| `physics_response_matrix.png` | full E_in → E_out response as a heatmap, for ideal / full / no-fluorescence / escape-only. The K-escape ridge and the fluorescence line are visible | `pctk_figs.py` |
| `physics_response_slices.png` | slices of the above at 60/100/120 keV — peak positions readable | `pctk_figs.py` |
| `physics_crosstalk_3x3.png` | counts landing in each of the 9 pixels, per energy window. Shows cross-talk is a bottom-window effect (52% off-centre in bin 1, ~0% in bins 3–4) | `pctk_figs.py` |
| `physics_window_ablation.png` | window counts for full / no-fluorescence / escape-only / centre-pixel, vs the ideal detector | `pctk_figs.py` |
| **`physics_before_after_charge_sharing.png`** | **before/after with the parameters printed on it.** The same ideal-vs-sharing comparison at four levels — response matrix, binned spectrum, sinogram, reconstruction — with the PcTK parameter block (r0, sigma_e, dpix, dz, thresholds, N0) in the header | `fig_before_after.py` |
| `physics_open_beam_spectrum.png` | open-beam (air) recorded spectrum, ideal vs PcTK — reproduces Fig 6(b) of Morovati et al. 2025 | `fig_open_beam_spectrum.py` |
| `physics_projection_views.png` | projection views per energy bin — reproduces Fig 6(a). Needs `--fov 60`, the head transmits 0.7% so the full-scale render is black | `fig_projection_views.py --fov 60` |

## cascade — one effect at a time

Order is: ideal → + charge sharing → + K-escape → + pile-up → + noise.

| file | shows | script |
|---|---|---|
| **`cascade_per_energy_bin.png`** | 9 rows, one per energy bin: ideal / cross-talk / +pile-up / relative error. The error ramps from −87% in bin 1 to +67% in bin 9 | `fig_perbin.py` |
| `cascade_absolute.png` | sinogram and reconstruction at each stage, absolute | `fig_cascade.py` |
| `cascade_spectra.png` | recorded spectrum at each stage, three rays | `fig_cascade.py` |
| `cascade_delta_percent_bin{1,4}.png` | **use this one.** What each stage adds, as a % — shows the effect permeates the whole object uniformly | `fig_cascade_rel.py --bin 1` |
| `cascade_delta_counts_bin{1,4}.png` | same in absolute counts. **Misleading on its own** — counts span five decades so the air gap dominates and the effect looks confined to the detector edges. Kept only as a contrast to the percent version | `fig_cascade_diff.py --bin 1` |

## signal — counts per bin per stage

| file | shows | script |
|---|---|---|
| `signal_by_stage.png` | spectrum at each stage, six operating points from air to 20 cm + 6 cm bone | `signal_diff.py` |
| `signal_delta_per_stage.png` | what each effect adds as a % of the stage before it | `signal_diff.py` |

## data — the generated dataset

| file | shows | script |
|---|---|---|
| `data_phantom.png` | the two-material Shepp-Logan head | `make_paper_figs.py` |
| `data_sinogram_9bin.png` | line integrals, then count sinograms per bin | `make_paper_figs.py` |
| `data_signal_at_rays.png` | recorded spectrum at air / skull edge / centre | `make_paper_figs.py` |
| `data_sinogram_counts_4bin.png` | raw count sinograms, mean vs noisy | `pctk_view.py` |
| `data_line_integrals.png` | line integrals per window | `pctk_view.py` |
| `data_noise_qa.png` | noise check: z-score histogram, mean 0.0002 sd 0.9998 | `pctk_view.py` |

## recon — images

| file | shows | script |
|---|---|---|
| `recon_9bin_ideal_vs_pctk.png` | per-bin FBP, ideal vs PcTK. Bin 1 loses its low-energy character entirely | `make_paper_figs.py` |
| `recon_fbp_4bin.png` | FBP of the MATLAB-generated sinogram, mean and noisy | `pctk_recon.py` |
| `recon_pctk_authors_reference.png` | the PcTK authors' own reconstruction from `5_refdata` — what the phantom should look like | `pctk_view.py` |
| `recon_decomposition_wrong_vs_right.png` | decomposing the same counts with the ideal vs the PcTK model. **STALE — `demo_end2end.py` currently OOMs, so regenerate before trusting this one** | `demo_end2end.py` (broken) |

## stats

| file | shows | script |
|---|---|---|
| `stats_crlb_efficiency.png` | ML decomposition vs the Cramér–Rao bound. Efficiency 0.95–0.96 below ~3 cm bone (no headroom); above that "efficiency > 1" is severe bias, not a better estimator | `crlb_sweep.py` |

---

## Which to show for what

**"Charge sharing is real"** → `physics_charge_sharing_evidence.png`, on its own.

**"Here is the data before and after, and the parameters"** → `physics_before_after_charge_sharing.png`, on its own.

### `figures/before_after/` — the same 17 panels, one PNG each

`fig_before_after.py` writes the combined sheet *and* every panel separately, for
slides where one panel per slide is better than a wall of subplots.

Grayscale sense: **dark = more** (more counts, higher mu) in the sinogram and
reconstruction panels. Pass `--bright-is-more` for the matplotlib default instead.
The diverging difference panels are unaffected — `RdBu_r` is already dark at both
extremes and white at zero.

| file | shows |
|---|---|
| `00_parameters.png` | the PcTK parameter block, read from `SRF_param_v32_20170622.csv` |
| `01_response_ideal.png` / `02_response_sharing.png` | E_in → E_out response, before / after |
| `03_photon_100keV.png` | one 100 keV photon: line → smear |
| `04_energy_3x3.png` | where that photon's energy lands across the 3×3 neighbourhood |
| `05_bins_air.png` / `06_bins_soft.png` / `07_bins_bone.png` | binned counts before/after, three attenuation paths |
| `08_bin_shift.png` | % change per bin — the down-shift, all three paths on one axis |
| `09_sinogram_ideal.png` / `10_sinogram_sharing.png` / `11_sinogram_diff.png` | sinogram before / after / difference |
| `12_sinogram_profile.png` | one view through the head, both curves, with the after/before **ratio** overlaid as a dotted line — the offset is not constant, it runs 1.5x in air to 50x at the skull edge |
| **`17_sinogram_ratio_bins.png`** | **the volatility panel.** after/before ratio vs channel for all 4 bins. Bin 1 spikes to 50x at the skull edge; **bin 2 crosses 1.0**, losing counts in air and gaining through the object. Shows the distortion is path-dependent, not a constant scale factor |
| **`18_sinogram_profile_bin2.png`** | bin 2 profile where the two curves visibly **cross**: air -20%, object -14% to +51% |
| **`19_sinogram_profile_9bin.png`** | **all 9 bins** at Morovati's binning (20-109 keV, 10 keV steps), before vs after, one panel per bin. Bin 4 (50-60 keV) is the sign-flipper: -8% in air, +18% to +121% through the object | `fig_profile_9bin.py` |
| `20_sinogram_ratio_9bin.png` | all 9 ratios on one axis with the line at 1.0 | `fig_profile_9bin.py` |
| `bins9/profile_bin{1..9}.png` | the 9 panels of figure 19, one PNG each | `fig_profile_9bin.py` |
| **`21_profile_9bin_pileup.png`** | **the full cascade, all 9 bins.** Four curves per panel: ideal / + pile-up only / + charge sharing only / + both. Stage 2 is an ablation (pile-up on an ideal deposited spectrum) that isolates readout from sensor | `fig_profile_9bin_pileup.py --tau 30` |
| `bins9_pileup/profile_bin{1..9}.png` | the 9 panels of figure 21, one PNG each | `fig_profile_9bin_pileup.py` |

Figure 21 also writes `outputs/profile9_pileup.npz` with the four stages as arrays
(`ideal`, `pileup`, `sharing`, `both`, each 1854 x 9) plus the ray path lengths, so the
numbers can be re-plotted without rebuilding the grid (~12 min).

Key result: **pile-up is an air/low-attenuation effect, charge sharing is not.** In air
bin 9 gains +529% from sum events; through the head pile-up falls to 0-8% in bins 3-9 because
the rate drops ~300x. The two do not compose — bin 9 in air is +529% and -50% separately but
+419% together, since charge sharing changes the deposited spectrum that pile-up then acts on.
| `13_image_ideal.png` / `14_image_sharing.png` / `15_image_diff.png` | reconstruction before / after / difference |
| `16_image_profile.png` | central horizontal profile through the reconstruction |

**"Here is what each effect does"** → `cascade_per_energy_bin.png`, then `signal_delta_per_stage.png`.

**"Here is why a network can't fix it"** → `stats_crlb_efficiency.png`.

**Do not show** `cascade_delta_counts_*` without the percent version beside it — absolute count deltas are dominated by the air gap and give the wrong impression.

## Not yet built

The figure the project actually needs is **nominal vs empirical coverage on the HAP rods**, one line per arm. It requires a trained model and does not exist.

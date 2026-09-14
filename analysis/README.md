# PcTK analysis

Work built on the PcTK 3.2 photon-counting detector toolkit (`../PcTK_3.24a/`).
Everything here resolves its own paths, so scripts run from any working directory.
Keep `analysis/` beside `PcTK_3.24a/`.

## Layout

    src/        Python analysis (paths.py holds every location)
    matlab/     MATLAB runner + Statistics-Toolbox stand-ins
    cache/      extracted PcTK response tables (~9 MB, regenerated if deleted)
    figures/    generated PNGs
    outputs/    generated arrays

## Python

    python3 src/pctk_compare.py     # ablations + pileup cascade vs an ideal detector
    python3 src/pctk_figs.py        # fig1-fig4: response matrices, slices, 3x3 map, windows
    python3 src/pctk_view.py        # viewA-viewD: sinograms, line integrals, noise QA
    python3 src/pctk_recon.py       # viewE: fan-to-parallel rebinning + FBP  (~2 min)

`pctk_compare.py` has a `CFG` block at the top. Free toggles, no MATLAB needed:

| toggle | values |
|---|---|
| `FLUOR`  | `'full'` / `'off'` / `'escape_only'` |
| `PIXELS` | `'3x3'` / `'center'` |
| `ETH`    | any thresholds, any number of windows |
| `N0`, `PATHS`, `TAU_NS`, `PILEUP` | anything |

Changing `r_0`, `sigma_e`, `dpix`, `dz` instead requires regenerating in MATLAB.

## MATLAB

    run_pctk('nocorr')   % ~1 min
    run_pctk('corr')     % 6-7 h
    run_pctk('gennc')    % ~70 min, new detector parameters

This machine has **no Statistics and Machine Learning Toolbox**, so `matlab/`
supplies `mvnrnd`, `normrnd`, `random`, `poissrnd` in base MATLAB. Noise
realizations therefore come from these, not from MathWorks' generators
(verified: z-score of (noisy-mean)/sqrt(mean) has mean 0.0002, sd 0.9998).
Drop `matlab/` from the path if you run somewhere the toolbox exists.

## Verified facts

- The Python E-domain binning reproduces the shipped `nCov3x3w` to **3.3e-16**.
- Generated air counts `[1006437, 248121, 110108, 67305]` match the PcTK
  authors' own documented values (comment at `3_src/script_workflow_PcTK.m:304`)
  to 0.004%. The shipped `v_N0` = 1.41e6 is correct for reproducing the example.
- Using `m2_SRE_q0/q1/q2` requires two fixes the manual omits: fill pixels
  1,2,3,6,7,8 by quadrant symmetry from pixels 4,5,9, and zero `v_Eo > 175 keV`
  (an overflow accumulator). Then the manual's identity holds to 1.3e-4.

## Gotchas

- `5_refdata/` is **not** a validation target for the shipped input: it is 32 rows
  vs. 7, at a different flux.
- Both workflow scripts write the same output filenames.
- The final `.mat` save needs ~1.7 GB; it crashed MATLAB here after writing. The
  `.flt` files are written incrementally and survive regardless.
- `pctk_recon.py` is geometrically correct but its absolute mu scaling is unvalidated.

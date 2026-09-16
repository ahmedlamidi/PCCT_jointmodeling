"""Data for Morovati et al. 2025's OWN problem (doi 10.1088/1361-6560/adaf71).

A separate arm from generate_baseline_dataset3d.py (the head phantom). Nothing
here touches that arm, its data, or outputs/norm_stats.json.

WHAT IS MATCHED (their wording in quotes)
    phantom      "five ellipsoids of different materials fully enclosed within a
                 sphere" of water, air outside -- phantom_morovati.py
    materials    "soft tissue, adipose tissue, brain gray matter, white matter,
                 blood, and cortical bone" -- NIST ICRU-44, nist_materials.py
    size         "256 x 256 x 256 cube with a voxel size of 0.113 mm^3", read as
                 0.113 mm per side (28.9 mm cube). Their MARS detector's 0.11 mm
                 pixels support that reading; 0.113 mm^3 per voxel would be
                 0.48 mm per side.
    phantoms     "10 randomly created 3D material phantoms" for training,
                 "Another 5" for testing -> --ntrain 10 --ntest 5
    projections  180, parallel beam
    detector     PcTK 3.2, CdTe, thresholds 20-110 keV -> nine CLOSED bins
    patches      16 x 16 x 9, stride 8 on the detector plane. A 272-channel x
                 256-row plane gives 33 x 31 = 1023 patches per projection --
                 exactly their 1,841,400 / (10 x 180). Detector pitch = voxel,
                 so the 256-voxel cube sits in the middle 256 of 272 channels.
    label        noise-free ideal counts; Poisson noise on the input only
    pile-up      paralyzable (Bierme & Roessl 2012; Roessl et al 2016), tau 30 ns

WHAT IS NOT MATCHED, AND WHY
    pixel pitch  PcTK ships charge-sharing tables for 225 um pixels only. Here the
                 detector SAMPLES at 0.113 mm, but its physics is the 225 um
                 table, which under-states charge sharing for a smaller pixel.
                 Their pitch is not stated.
    flux         unstated in their paper. N0 = 12,000 photons/pixel/view at
                 4,000 views/s, the same per-pixel rate as the head arm, so
                 lam*tau in air is 2.45 in both.
    spectrum     theirs: SpekCalc, "12 to 120 keV"; ours: PcTK's 120 kVp + 2 mm Al
    angular arc  not stated. 180 views over 180 deg (--arc), the standard parallel
                 range; the head arm uses 360 deg.
    grey/white   one NIST entry, so identical here
    cross-talk   their modification inside the pile-up stage is not modelled

    python3 generate_morovati_dataset.py                                # full arm, ~4-5 h
    python3 generate_morovati_dataset.py --n 64 --nch 80 --nview 12 \\
            --ntrain 1 --ntest 1 --out morovati_smoke                   # minutes

NOT EXECUTED on HiPerGator. Run locally at reduced size only (see RUNBOOK).
"""
import argparse, json, os, time
import numpy as np
from scipy.ndimage import rotate
from paths import OUT
import pctk_compare as P
from pipeline import Forward
import nist_materials as NM
import phantom_morovati as PM
from generate_baseline_dataset3d import ETH9_CLOSED


def project(vols, nview, arc_deg, nch, vox_cm):
    """Parallel beam about z. vols (k, nz, n, n) -> (k, nview, nz, nch) line integrals, cm.

    Detector pitch = voxel size, so each detector row is one axial slice and the
    n-voxel object sits in the middle n of nch channels; the rest is air.
    """
    k, nz, n, _ = vols.shape
    pad = (nch - n) // 2
    flat = vols.reshape(k * nz, n, n)
    out = np.zeros((k, nview, nz, nch), np.float32)
    for v in range(nview):
        r = rotate(flat, v * arc_deg / nview, axes=(1, 2), reshape=False, order=1,
                   mode='constant', cval=0.0)
        out[:, v, :, pad:pad + n] = r.sum(axis=1).reshape(k, nz, n) * vox_cm
    return out


def grid_axes(coef, diam_cm, step):
    """Pile-up grid covering every (brain-eq, bone-eq) path this phantom family can give.

    brain-eq <= max(a) x sphere diameter. bone-eq runs from min(b) x diameter
    (adipose and water are slightly NEGATIVE bone) up to one diameter of bone.
    The negative row sits EXACTLY at that bound, not a whole step below it: a
    corner like (0 cm, -0.25 cm) is unphysical (net negative attenuation) and
    bent the interpolation next to it by 2-4% when it was used (measured).
    """
    amax = max(c[0] for c in coef.values())
    bmin = min(min(c[1] for c in coef.values()), 0.0)
    gb = np.arange(int(np.ceil(amax * diam_cm / step)) + 1) * step
    gc = np.arange(0.0, np.ceil(diam_cm / step) * step + step / 2, step)
    if bmin < 0:
        gc = np.concatenate([[bmin * diam_cm], gc])
    return gb, gc


def save_manifest(outdir, manifest):
    tmp = os.path.join(outdir, 'manifest.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, os.path.join(outdir, 'manifest.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ntrain', type=int, default=10)
    ap.add_argument('--nval', type=int, default=0,
                    help='they have no validation PHANTOMS (their 95/5 split is by patch, '
                         'which leaks). --nval 1 adds one, beyond their protocol.')
    ap.add_argument('--ntest', type=int, default=5)
    ap.add_argument('--n', type=int, default=256, help='cube side, voxels (= detector rows)')
    ap.add_argument('--vox_mm', type=float, default=0.113)
    ap.add_argument('--nch', type=int, default=272, help='detector channels (>= n, same parity)')
    ap.add_argument('--nview', type=int, default=180)
    ap.add_argument('--arc', type=float, default=180.0, help='angular range, degrees')
    ap.add_argument('--sphere_r', type=float, default=0.9, help='fraction of the cube half-width')
    ap.add_argument('--axis_lo', type=float, default=0.15, help='ellipsoid semi-axis / sphere radius')
    ap.add_argument('--axis_hi', type=float, default=0.45)
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--tau', type=float, default=30.0, help='dead time, ns (0 = no pile-up)')
    ap.add_argument('--t_p', type=float, default=10.0, help='pulse peaking time, ns')
    ap.add_argument('--T', type=float, default=25.0, help='pulse length, ns; must be <= tau')
    ap.add_argument('--pileup_mode', default='paralyzable',
                    choices=['paralyzable', 'seminonparalyzable'])
    ap.add_argument('--grid_step', type=float, default=0.25,
                    help='pile-up grid spacing, cm. The head arm uses 1-2 cm; a 2.6 cm '
                         'object needs finer steps (bin 9 pile-up moves +197%% -> +144%% '
                         'between 0 and 2.9 cm of water).')
    ap.add_argument('--maxrays', type=int, default=300_000,
                    help='rays per forward-model chunk; peak array ~170*maxrays*8 bytes')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='morovati_match')
    a = ap.parse_args()

    if (a.nch - a.n) % 2 or a.nch < a.n:
        raise SystemExit('--nch must be >= --n with the same parity (object centred)')
    if a.tau > 0 and a.T > a.tau:
        raise SystemExit('refusing to run: pulse length T=%.0f ns exceeds dead time '
                         'tau=%.0f ns, so every pulse self-retriggers.' % (a.T, a.tau))
    SPLITS = [('train', a.ntrain), ('val', a.nval), ('test', a.ntest)]
    total = a.ntrain + a.nval + a.ntest
    print('phantoms: %d train / %d val / %d test' % (a.ntrain, a.nval, a.ntest))

    # ---- materials: NIST ICRU-44, two-basis ---------------------------------
    coef = NM.two_basis()
    print('materials as a*brain + b*cortical bone (NIST ICRU-44, fit 20-120 keV):')
    for k, (ca, cb, err) in coef.items():
        print('  %-12s a %.3f  b %+.3f  max err %.2f%%' % (k, ca, cb, err))

    vox_cm = a.vox_mm / 10.0
    diam_cm = 2 * a.sphere_r * (a.n / 2) * vox_cm
    gb, gc = grid_axes(coef, diam_cm, a.grid_step)
    print('object: %d^3 voxels x %.3f mm = %.1f mm cube, sphere %.1f mm across'
          % (a.n, a.vox_mm, a.n * a.vox_mm, diam_cm * 10))
    print('pile-up grid: brain-eq %.2f..%.2f cm (%d)  x  bone-eq %.2f..%.2f cm (%d)  = %d points'
          % (gb[0], gb[-1], len(gb), gc[0], gc[-1], len(gc), len(gb) * len(gc)), flush=True)

    # ---- forward model on the NIST basis ------------------------------------
    cfg = dict(P.CFG)
    cfg['ETH'] = ETH9_CLOSED
    # Monte Carlo settings are the head arm's (Forward defaults). Measured on this
    # grid: interpolated vs direct MC on real rays (water, adipose+water, blood,
    # bone, grazing) agree to <= 2.2%, sign alternating across bins -- MC noise,
    # not interpolation. The noise is set by mc_pulse_train's 400k-pulse cap
    # (~150 frames near air), so asking for more frames changes nothing.
    F = Forward(cfg=cfg, tau_ns=a.tau, t_p_ns=a.t_p, T_ns=a.T, pileup_mode=a.pileup_mode,
                mu=NM.basis(), grid_brain=gb, grid_bone=gc)
    Nl = F.Nl - 1                                  # drop the >=110 keV overflow bin
    _, _, air_i = F.mean_cov(np.zeros(1), np.zeros(1))
    air_counts = [float(v) for v in air_i[0, :Nl]]

    npv = (1 + (a.nch - a.patch) // a.stride) * (1 + (a.n - a.patch) // a.stride)
    tot = npv * a.nview * total
    print('detector plane %d ch x %d rows, %d views over %g deg' % (a.nch, a.n, a.nview, a.arc))
    print('patches: %d per projection, %d total   (theirs: 1023 / 1,841,400 for 10 phantoms)'
          % (npv, tot), flush=True)
    if npv <= 0:
        raise SystemExit('0 patches per projection: plane smaller than the patch')

    outdir = os.path.join(OUT, a.out)
    for nm, cnt in SPLITS:
        if cnt:
            os.makedirs(os.path.join(outdir, nm), exist_ok=True)
    manifest = dict(
        arm_kind='morovati_phantom', eth=list(cfg['ETH'][:Nl]), label='clean',
        air_counts=air_counts, closed_top_bin=True, bins=Nl,
        nview=a.nview, arc_deg=a.arc, nch=a.nch, nrow=a.n, npix=a.n, vox_mm=a.vox_mm,
        patch=a.patch, stride=a.stride, patches_per_projection=npv, patches_total=tot,
        tau_ns=a.tau, t_p_ns=a.t_p, T_ns=a.T, N0=cfg['N0'], seed=a.seed,
        pileup_mode=(a.pileup_mode if a.tau > 0 else 'none (tau=0)'),
        pileup_grid=dict(brain=gb.tolist(), bone=gc.tolist(),
                         mc='Forward defaults (250 frames, 400k-pulse cap); grid values '
                            'carry ~1-2% MC noise near air (measured 2026-09-14)'),
        geometry='parallel beam about z, detector pitch = voxel, one row per axial slice',
        basis='NIST ICRU-44 brain + cortical bone (nist_materials.py), NOT PcTK m2_mukE.csv',
        materials={k: dict(a=v[0], b=v[1], max_err_pct=v[2], density=NM.NIST[k][1])
                   for k, v in coef.items()},
        phantom=dict(sphere_r=a.sphere_r, axis_lo=a.axis_lo, axis_hi=a.axis_hi, nell=5),
        matched='their phantom family and materials, 256^3 x 0.113 mm, 10+5 phantoms, 180 '
                'parallel projections, PcTK 3.2, 9 closed bins, 16x16x9 stride 8 (1023 per '
                'projection), noise-free label, paralyzable pile-up',
        unmatched='pixel pitch (PcTK 225 um tables, detector samples 0.113 mm); flux '
                  '(unstated); spectrum (SpekCalc vs PcTK 120 kVp); arc (unstated); grey = '
                  'white matter (one NIST entry); cross-talk inside pile-up',
        n_train=a.ntrain, n_val=a.nval, n_test=a.ntest, split_level='phantom',
        train=[], val=[], test=[], phantoms={})

    owner = sum(([nm] * cnt for nm, cnt in SPLITS), [])
    first, c = {}, 0
    for nm, cnt in SPLITS:
        first[nm] = c
        c += cnt
    rng_ph = np.random.default_rng(a.seed)         # phantom shapes
    rng = np.random.default_rng(a.seed + 1)        # detector noise
    chunk = max(1, a.maxrays // (a.n * a.nch))
    t0 = time.time()

    for i in range(total):
        split = owner[i]
        k = i - first[split]
        lab, ells = PM.random_phantom(a.n, rng_ph, a.sphere_r, 5, a.axis_lo, a.axis_hi)
        A, B = PM.equivalent_maps(lab, coef)
        del lab
        SB, SBO = project(np.stack([A, B]), a.nview, a.arc, a.nch, vox_cm)   # (nview, nrow, nch)
        del A, B
        # Forward._interp CLAMPS paths outside the grid -- silently wrong counts. Refuse.
        if SB.max() > gb[-1] or SBO.max() > gc[-1] or SBO.min() < gc[0]:
            raise SystemExit('path outside the pile-up grid: brain-eq max %.3f (grid %.2f), '
                             'bone-eq %.3f..%.3f (grid %.2f..%.2f)'
                             % (SB.max(), gb[-1], SBO.min(), SBO.max(), gc[0], gc[-1]))

        X = np.zeros((a.nview, a.n, a.nch, Nl), np.float32)
        Y = np.zeros_like(X)
        MD = np.zeros_like(X)
        for s in range(0, a.nview, chunk):
            e = min(s + chunk, a.nview)
            yd, _, md, mi = F.sample(SB[s:e].reshape(-1), SBO[s:e].reshape(-1), rng)
            sh = (e - s, a.n, a.nch, Nl)
            X[s:e] = yd[:, :Nl].reshape(sh)
            Y[s:e] = mi[:, :Nl].reshape(sh)        # noise-free ideal label
            MD[s:e] = md[:, :Nl].reshape(sh)

        V = np.stack([SB, SBO], -1)
        fn = '%s/phantom_%03d.npz' % (split, k)
        np.savez_compressed(os.path.join(outdir, fn), X=X, Y=Y, MU_D=MD, V=V)
        manifest[split].append(fn)
        manifest['phantoms'][fn] = ells
        save_manifest(outdir, manifest)            # after every phantom: a partial run is usable
        print('  %-5s %2d  %s  brain-eq<=%.2f cm  bone-eq %+.2f..%.2f cm  X%s  %.0fs'
              % (split, k + 1, '/'.join(sorted(e_['material'] for e_ in ells)),
                 SB.max(), SBO.min(), SBO.max(), X.shape, time.time() - t0), flush=True)
        del X, Y, MD, V, SB, SBO

    print('\nwrote %s  (%.0fs)' % (outdir, time.time() - t0))
    print('normalisation NOT built: outputs/norm_stats.json belongs to the head arm.')


if __name__ == '__main__':
    main()

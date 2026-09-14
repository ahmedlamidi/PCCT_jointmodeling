"""Closest achievable match to Morovati et al. 2025 (doi 10.1088/1361-6560/adaf71).

WHAT IS MATCHED
    10 training + 5 test phantoms, 3D Shepp-Logan variants
    180 projections per phantom
    9 energy bins, 20-109 keV in 10 keV steps
    16 x 16 x 9 patches taken from the DETECTOR PLANE (u, v), stride 8
    ~1.8M patches
    FBP-compatible fan geometry

WHY (u, v) AND STRIDE 8
    1,841,400 patches / (10 phantoms x 180 projections) = exactly 1023 per
    projection. With 16x16 patches that is consistent with a ~256 x 272 detector
    plane at stride 8. So they patch the detector plane, not the sinogram, and
    the stride is 8. Both are adopted here.

WHAT CANNOT BE MATCHED (their methods section was not reachable)
    detector model and its parameters   -> PcTK, 225 um pitch, 1600 um CdTe
    incident spectrum / kVp             -> PcTK's 120 kVp (their top bin at
                                           109 keV implies ~110-120 kVp)
    flux, dead time                     -> our choice; tau=0 by default
    ground-truth label definition       -> ours: same spectrum, perfect
                                           thresholds, same QE, Poisson, SAME SEED
    cone beam                           -> multi-slice fan (one detector row per
                                           axial slice). Valid at small cone angle.

    The label definition is the one that matters most. If theirs is a noise-free
    reference or a high-dose acquisition rather than an ideal detector, the two
    tasks differ and numbers are not comparable even in principle.

    python3 generate_baseline_dataset3d.py --ntrain 10 --ntest 5
"""
import argparse, json, os, time
import numpy as np
from paths import OUT
import pctk_compare as P
from pipeline import Forward
from projector import Geometry
from phantom import variants3d

ETH9 = [20., 30., 40., 50., 60., 70., 80., 90., 100.]
# Morovati use thresholds 20,30,...,110 keV giving nine CLOSED bins 20-29 ... 100-109.
# Ours above leaves bin 9 open (100-inf), which collects pile-up sum events that
# their top threshold discards -- a real difference at high count rate.
ETH9_CLOSED = ETH9 + [110.]


def main():
    ap = argparse.ArgumentParser()
    # Split is by PHANTOM, never by patch: patches from one phantom are highly
    # correlated, so a patch-level split leaks and inflates validation scores.
    ap.add_argument('--nphantom', type=int, default=20,
                    help='total phantoms, divided by --split. 20 makes 70/15/15 exact '
                         '(14/3/3). Morovati et al. use 10 train + 5 test and no '
                         'validation split; pass --ntrain/--ntest to reproduce that.')
    ap.add_argument('--split', type=int, nargs=3, default=[70, 15, 15],
                    metavar=('TRAIN', 'VAL', 'TEST'),
                    help='percentages, must sum to 100')
    ap.add_argument('--ntrain', type=int, default=None,
                    help='override: explicit phantom counts, disables --split')
    ap.add_argument('--nval', type=int, default=None)
    ap.add_argument('--ntest', type=int, default=None)
    ap.add_argument('--nview', type=int, default=180)
    ap.add_argument('--npix', type=int, default=256)      # in-plane phantom size
    ap.add_argument('--nrow', type=int, default=64)       # detector rows = axial slices
    ap.add_argument('--nch', type=int, default=1854)      # PcTK native
    ap.add_argument('--crop', type=int, default=512,      # channels kept around centre
                    help='channels retained about the detector centre (object support)')
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--tau', type=float, default=0.0, help='dead time, ns (0 = no pile-up)')
    ap.add_argument('--t_p', type=float, default=10.0, help='pulse peaking time, ns')
    ap.add_argument('--T', type=float, default=25.0,
                    help='pulse length, ns. MUST be <= tau or every pulse '
                         'retriggers on its own tail and bin 1 inflates ~38%%')
    # Chunk on RAYS, not views. One view here is nrow*crop rays (59k at 32x1854),
    # and Forward.sample builds a (170, nrays) float64 spectrum -- 170*nrays*8
    # bytes. Chunking on views blew past 4 GB on the first chunk.
    ap.add_argument('--maxrays', type=int, default=300_000,
                    help='rays per forward-model chunk; peak array is ~170*maxrays*8 bytes')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label', default='noisy', choices=['noisy', 'clean'],
                    help="'clean' saves the NOISE-FREE ideal mean as the label, which is "
                         "what Morovati et al. use ('Poisson noise was added' to the "
                         "distorted data only). 'noisy' keeps the old shared-seed label.")
    ap.add_argument('--geometry', default='parallel', choices=['parallel', 'fan'],
                    help="Morovati et al. simulate '180 spectral projections ... in a "
                         "parallel beam configuration', so 'parallel' is the default. "
                         "'fan' uses the R=600/Rd=1080 fan geometry.")
    ap.add_argument('--pileup_mode', default='paralyzable',
                    choices=['paralyzable', 'seminonparalyzable'],
                    help="'paralyzable' (Bierme & Roessl 2012, Roessl et al 2016) is what "
                         "Morovati et al. use. 'seminonparalyzable' is Yang et al. 2025. "
                         "They agree below lam*tau ~ 0.1 and differ 3.4x at lam*tau = 2.45.")
    ap.add_argument('--closed_top_bin', action='store_true',
                    help='add a 110 keV threshold and discard counts above it, giving nine '
                         'CLOSED bins 20-29 ... 100-109 as in Morovati et al.')
    ap.add_argument('--out', default='baseline3d')
    a = ap.parse_args()

    # ---- resolve the split -------------------------------------------------
    if a.ntrain is not None or a.ntest is not None:
        ntr = a.ntrain or 0; nva = a.nval or 0; nte = a.ntest or 0
    else:
        if sum(a.split) != 100:
            raise SystemExit('--split must sum to 100, got %s' % (a.split,))
        n = a.nphantom
        ntr = round(n * a.split[0] / 100)
        nva = round(n * a.split[1] / 100)
        nte = n - ntr - nva                      # remainder, so the total is exact
    if min(ntr, nte) < 1:
        raise SystemExit('need at least 1 train and 1 test phantom, got %d/%d/%d'
                         % (ntr, nva, nte))
    total = ntr + nva + nte
    SPLITS = [('train', ntr), ('val', nva), ('test', nte)]
    print('phantoms: %d train / %d val / %d test  (%d total, %.0f/%.0f/%.0f%%)'
          % (ntr, nva, nte, total, 100*ntr/total, 100*nva/total, 100*nte/total))

    cfg = dict(P.CFG)
    cfg['ETH'] = ETH9_CLOSED if a.closed_top_bin else ETH9
    outdir = os.path.join(OUT, a.out)
    for nm, cnt in SPLITS:
        if cnt:
            os.makedirs(os.path.join(outdir, nm), exist_ok=True)

    print('building forward model (9 bins)...', flush=True)
    if a.tau > 0 and a.T > a.tau:
        raise SystemExit('refusing to run: pulse length T=%.0f ns exceeds dead time '
                         'tau=%.0f ns, so every pulse self-retriggers.' % (a.T, a.tau))
    F = Forward(cfg=cfg, tau_ns=a.tau, t_p_ns=a.t_p, T_ns=a.T,
                pileup_mode=a.pileup_mode)
    g = Geometry(nview=a.nview, nch=a.nch, npix=a.npix)
    Nl = F.Nl - 1 if a.closed_top_bin else F.Nl     # drop the >=110 keV overflow bin
    # Physical ceiling on any count: the ideal, UNATTENUATED (air) count per bin,
    # straight from the forward model at zero path length. Written to the manifest
    # so the loader can clip outputs to it -- exact, and independent of whether the
    # crop happens to contain air rays. mu_i is never touched by pile-up.
    _, _, air_i = F.mean_cov(np.zeros(1), np.zeros(1))
    air_counts = [float(v) for v in air_i[0, :Nl]]
    print('label=%s  bins=%d%s' % (a.label, Nl,
          '  (closed top bin, 100-109 keV)' if a.closed_top_bin else '  (open top bin, 100-inf)'))
    c0 = a.nch // 2 - a.crop // 2
    sl = slice(c0, c0 + a.crop)

    chunk = max(1, a.maxrays // (a.nrow * a.crop))
    print('chunk = %d views (%d rays, peak spectrum array ~%.0f MB)'
          % (chunk, chunk * a.nrow * a.crop, 170 * chunk * a.nrow * a.crop * 8 / 1e6))

    npv = ((1 + (a.crop - a.patch) // a.stride) *
           (1 + (a.nrow - a.patch) // a.stride))
    tot = npv * a.nview * total
    print('detector plane %d ch x %d rows,  %d views' % (a.crop, a.nrow, a.nview))
    print('patches: %d per projection, %d total   (theirs: 1023 / 1,841,400)'
          % (npv, tot), flush=True)
    if npv <= 0:
        raise SystemExit('0 patches per projection: the detector plane (%d ch x %d rows) '
                         'is smaller than the %dx%d patch. Raise --nrow/--crop or lower '
                         '--patch.' % (a.crop, a.nrow, a.patch, a.patch))

    manifest = dict(eth=list(cfg['ETH'][:Nl]), label=a.label, air_counts=air_counts,
                    closed_top_bin=bool(a.closed_top_bin), nview=a.nview, nch=a.crop, nrow=a.nrow, npix=a.npix,
                    bins=Nl, patch=a.patch, stride=a.stride, tau_ns=a.tau,
                    t_p_ns=a.t_p, T_ns=a.T, seed=a.seed, patches_per_projection=npv, patches_total=tot,
                    N0=cfg['N0'],
                    geometry='multi-slice %s beam, one row per axial slice' % a.geometry,
                    pileup_mode=(a.pileup_mode if a.tau > 0 else 'none (tau=0)'),
                    matched='PcTK 3.2 detector, phantoms, projections, bins, patch size, '
                            'stride, (u,v) patching, noise-free label, closed top bin, '
                            'parallel beam, paralyzable pile-up',
                    unmatched='spectrum (PcTK shipped 120 kVp vs their SpekCalc 12-120 keV); '
                              'flux and dead time (unstated in their paper); spatial '
                              'cross-talk inside the pile-up stage (their modification)',
                    split_pct=a.split, n_train=ntr, n_val=nva, n_test=nte,
                    split_level='phantom',
                    train=[], val=[], test=[])

    rng = np.random.default_rng(a.seed + 1)
    # phantom index -> split name, in order
    owner = sum(([nm] * cnt for nm, cnt in SPLITS), [])
    first = {}
    c = 0
    for nm, cnt in SPLITS:
        first[nm] = c; c += cnt
    t0 = time.time()

    for i, (brain, bone) in enumerate(variants3d(a.npix, a.nrow, total, seed=a.seed)):
        split = owner[i]
        k = i - first[split]

        # project every axial slice onto its own detector row
        SB = np.empty((a.nrow, a.crop, a.nview), np.float32)
        SBO = np.empty_like(SB)
        proj = g.forward_parallel if a.geometry == 'parallel' else g.forward
        for r in range(a.nrow):
            SB[r] = proj(brain[r])[sl]
            SBO[r] = proj(bone[r])[sl]

        X = np.zeros((a.nview, a.nrow, a.crop, Nl), np.float32)
        Y = np.zeros_like(X); MD = np.zeros_like(X)
        for s in range(0, a.nview, chunk):
            e = min(s + chunk, a.nview)
            vb = SB[:, :, s:e].reshape(-1)
            vbo = SBO[:, :, s:e].reshape(-1)
            yd, yi, md, mi = F.sample(vb, vbo, rng)
            if a.closed_top_bin:                    # discard the overflow bin
                yd, yi, md, mi = yd[:, :Nl], yi[:, :Nl], md[:, :Nl], mi[:, :Nl]
            lab = mi if a.label == 'clean' else yi
            sh = (a.nrow, a.crop, e - s, Nl)
            X[s:e] = yd.reshape(sh).transpose(2, 0, 1, 3)
            Y[s:e] = lab.reshape(sh).transpose(2, 0, 1, 3)
            MD[s:e] = md.reshape(sh).transpose(2, 0, 1, 3)

        V = np.stack([SB.transpose(2, 0, 1), SBO.transpose(2, 0, 1)], -1).astype(np.float32)
        fn = '%s/phantom_%03d.npz' % (split, k)
        np.savez_compressed(os.path.join(outdir, fn), X=X, Y=Y, MU_D=MD, V=V)
        manifest[split].append(fn)
        print('  %-5s %2d  soft<=%.1f cm bone<=%.1f cm  X%s  %.0fs'
              % (split, k + 1, SB.max(), SBO.max(), X.shape, time.time() - t0), flush=True)

    with open(os.path.join(outdir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print('\nwrote %s  (%.0fs)' % (outdir, time.time() - t0))


if __name__ == '__main__':
    main()

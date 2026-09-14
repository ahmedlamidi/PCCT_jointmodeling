"""Training data matching Morovati et al.'s setup, on our generator.

Their simulated setup (Phys Med Biol 2025, doi 10.1088/1361-6560/adaf71):
    10 training phantoms, 5 test, Shepp-Logan variants
    180 spectral projections per phantom
    9 energy bins, 20-109 keV in 10 keV steps
    16 x 16 x 9 patches, 95/5 train/val

Reproduced here with PcTK physics instead of their detector model. Everything
except the detector is matched.

SINOGRAMS ARE SAVED, NOT PATCHES. 1.8M patches of 16x16x9 float32 would be
~17 GB; the sinograms they come from are ~24 MB per phantom. Patch in the
dataloader, which is also what your README asks for ("generate training samples
on the fly in the dataloader, not pre-written to disk").

Patches are taken in (channel, view) space -- for 2D fan beam the sinogram IS
the projection-domain image. Their detector had enough rows for 16x16 in
(u, v); PcTK's shipped geometry has only 7 rows, so that is not available.

    python3 generate_baseline_dataset.py --ntrain 10 --ntest 5
    python3 generate_baseline_dataset.py --ntrain 40 --nview 180   # more patches

Each phantom file holds:
    X      (nview, nch, Nl) float32  distorted counts   -- network input
    Y      (nview, nch, Nl) float32  ideal counts       -- label
    MU_D   (nview, nch, Nl) float32  distorted mean     -- noise-free target
    V      (nview, nch, 2)  float32  true path lengths, cm
    brain, bone  (npix, npix) float32  the phantom itself
"""
import argparse, json, os, time
import numpy as np
from paths import OUT
import pctk_compare as P
from pipeline import Forward
from projector import Geometry
from phantom import variants

ETH9 = [20., 30., 40., 50., 60., 70., 80., 90., 100.]
ETH4 = [20., 50., 65., 80.]


def patch_count(nch, nview, patch, stride):
    return (1 + (nch - patch) // stride) * (1 + (nview - patch) // stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ntrain', type=int, default=10)
    ap.add_argument('--ntest', type=int, default=5)
    ap.add_argument('--nview', type=int, default=180)      # theirs
    ap.add_argument('--npix', type=int, default=256)       # theirs (in-plane)
    ap.add_argument('--nch', type=int, default=1854)       # PcTK detector
    ap.add_argument('--bins', type=int, choices=[4, 9], default=9)
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--tau', type=float, default=0.0, help='pile-up dead time, ns')
    ap.add_argument('--chunk', type=int, default=60, help='views per forward-model chunk')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='baseline_ds')
    a = ap.parse_args()

    eth = ETH9 if a.bins == 9 else ETH4
    cfg = dict(P.CFG); cfg['ETH'] = eth
    outdir = os.path.join(OUT, a.out)
    for s in ('train', 'test'):
        os.makedirs(os.path.join(outdir, s), exist_ok=True)

    print('building forward model (%d bins)...' % a.bins, flush=True)
    F = Forward(cfg=cfg, tau_ns=a.tau)
    g = Geometry(nview=a.nview, nch=a.nch, npix=a.npix)
    Nl = F.Nl
    t0 = time.time()

    npp = patch_count(a.nch, a.nview, a.patch, a.stride)
    print('geometry %d ch x %d views x %d bins   -> %d patches/phantom, %d total'
          % (a.nch, a.nview, Nl, npp, npp * (a.ntrain + a.ntest)), flush=True)

    manifest = dict(eth=eth, nview=a.nview, nch=a.nch, npix=a.npix, bins=Nl,
                    patch=a.patch, stride=a.stride, tau_ns=a.tau, seed=a.seed,
                    patches_per_phantom=npp, N0=cfg['N0'],
                    matched_to='Morovati et al. 2025 (bins, projections, phantom family)',
                    detector='PcTK 3.2, 225 um pitch, 1600 um CdTe (NOT their detector)',
                    train=[], test=[])

    total = a.ntrain + a.ntest
    gen = variants(a.npix, count=total, seed=a.seed)
    rng = np.random.default_rng(a.seed + 1)

    for i, (brain, bone) in enumerate(gen):
        split = 'train' if i < a.ntrain else 'test'
        k = i if i < a.ntrain else i - a.ntrain

        sb = g.forward(brain); sbo = g.forward(bone)          # (nch, nview) cm
        X = np.zeros((a.nview, a.nch, Nl), np.float32)
        Y = np.zeros_like(X); MD = np.zeros_like(X)
        for s in range(0, a.nview, a.chunk):
            e = min(s + a.chunk, a.nview)
            yd, yi, md, _ = F.sample(sb[:, s:e].ravel(), sbo[:, s:e].ravel(), rng)
            X[s:e] = yd.reshape(a.nch, e - s, Nl).transpose(1, 0, 2)
            Y[s:e] = yi.reshape(a.nch, e - s, Nl).transpose(1, 0, 2)
            MD[s:e] = md.reshape(a.nch, e - s, Nl).transpose(1, 0, 2)

        V = np.stack([sb.T, sbo.T], -1).astype(np.float32)    # (nview, nch, 2)
        fn = '%s/phantom_%03d.npz' % (split, k)
        np.savez_compressed(os.path.join(outdir, fn), X=X, Y=Y, MU_D=MD, V=V,
                            brain=brain.astype(np.float32), bone=bone.astype(np.float32))
        manifest[split].append(fn)
        print('  %-5s %3d/%d  brain<=%.1f cm bone<=%.1f cm  %.0fs'
              % (split, k + 1, a.ntrain if split == 'train' else a.ntest,
                 sb.max(), sbo.max(), time.time() - t0), flush=True)

    with open(os.path.join(outdir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print('\nwrote %s  (%d train, %d test, %.0fs)'
          % (outdir, a.ntrain, a.ntest, time.time() - t0))


if __name__ == '__main__':
    main()

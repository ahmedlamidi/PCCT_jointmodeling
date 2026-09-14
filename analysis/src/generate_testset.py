"""Frozen evaluation set, swept over the operator axes.

Your README: "pre-generate and FREEZE a fixed evaluation set, so every coverage
number in the paper comes from the same data." This does that.

Two independent mismatch axes:

    rho   spatial charge sharing   (sensor)    -- mode 'spatial'
    tau   pile-up dead time, ns    (readout)   -- mode 'pileup'

One shared standard-normal draw z is reused across every operator setting in a
run, so a difference between two settings is physics, never dice. Ground-truth
path lengths V are identical across settings by construction.

    python3 generate_testset.py --mode spatial --rhos 0,0.25,0.5,1.0
    python3 generate_testset.py --mode pileup  --taus 0,5,10,20,40

Held-out views default to 1800:2000, disjoint from the 0:200 training default.

LIMITATION -- the two axes are not currently composable. rho stamps BINNED
counts over the 3x3 neighbourhood; pile-up acts on a pixel's deposited-ENERGY
spectrum. Doing both at once rigorously means stamping in the energy domain
(191 bins x 9 pixels), thresholding after pile-up. Sweep one axis at a time
until that is built.
"""
import argparse, json, os, time
import numpy as np
import scipy.io
from paths import PCTK, OUT
from pipeline import Forward
import pctk_compare as P


def parse_list(s):
    return [float(x) for x in s.split(',') if x.strip() != '']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['spatial', 'pileup'], default='spatial')
    ap.add_argument('--rhos', default='0,0.25,0.5,1.0', help='sharing strengths (mode spatial)')
    ap.add_argument('--taus', default='0,5,10,20,40', help='dead times in ns (mode pileup)')
    ap.add_argument('--views', default='1800:2000', help='held-out view slice')
    ap.add_argument('--row', type=int, default=3)
    ap.add_argument('--all-rows', action='store_true', help='use all 7 rows (spatial mode)')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', default='testset')
    a = ap.parse_args()

    v0, v1 = (int(x) for x in a.views.split(':'))
    outdir = os.path.join(OUT, a.out)
    os.makedirs(outdir, exist_ok=True)

    sv = scipy.io.loadmat(PCTK + '/1_inputdata/m4_sino_v.mat')['m4_sino_v']
    Nch, Nrow_all = sv.shape[1], sv.shape[2]
    rows = np.arange(Nrow_all) if (a.all_rows and a.mode == 'spatial') else np.array([a.row])
    nv = v1 - v0

    # ground truth, identical for every operator setting
    V = np.empty((nv, len(rows), Nch, 2), np.float32)
    for k, iv in enumerate(range(v0, v1)):
        V[k, :, :, 0] = sv[0, :, rows, iv].T if len(rows) > 1 else sv[0, :, rows[0], iv]
        V[k, :, :, 1] = sv[1, :, rows, iv].T if len(rows) > 1 else sv[1, :, rows[0], iv]

    manifest = dict(mode=a.mode, views=[v0, v1], rows=rows.tolist(), seed=a.seed,
                    phantom='PcTK m4_sino_v (brain/bone head)', files=[])
    np.save(os.path.join(outdir, 'V_truth.npy'), V)

    t0 = time.time()

    if a.mode == 'spatial':
        from noise_spatial import SpatialNoise
        F = Forward(tau_ns=0.0, verbose=False)
        rng = np.random.default_rng(a.seed)
        # one shared draw per view, reused for every rho
        Z = [rng.standard_normal((Nch * len(rows), 36)) for _ in range(nv)]
        for rho in parse_list(a.rhos):
            sn = SpatialNoise(F, rho=rho, verbose=False)
            X = np.zeros((nv, len(rows), Nch, F.Nl), np.float32)
            MU = np.zeros_like(X)
            for k in range(nv):
                vb = V[k, :, :, 0].T          # (Nch, Nrow)
                vbo = V[k, :, :, 1].T
                y = sn.sample_view(vb, vbo, rng, z=Z[k])
                m = sn.sample_view(vb, vbo, rng, noise_free=True)
                X[k] = y.transpose(2, 1, 0)   # -> (Nrow, Nch, Nl)
                MU[k] = m.transpose(2, 1, 0)
            fn = 'rho%0.2f.npz' % rho
            np.savez_compressed(os.path.join(outdir, fn), X=X, MU=MU, rho=rho)
            manifest['files'].append(dict(file=fn, rho=rho))
            print('  rho=%.2f  %s  %.0fs' % (rho, X.shape, time.time() - t0), flush=True)

    else:
        for tau in parse_list(a.taus):
            F = Forward(tau_ns=tau, verbose=False)
            rng = np.random.default_rng(a.seed)   # same seed per setting -> same z
            X = np.zeros((nv, len(rows), Nch, F.Nl), np.float32)
            MU = np.zeros_like(X); Y = np.zeros_like(X)
            for k in range(nv):
                vb = V[k, :, :, 0].ravel(); vbo = V[k, :, :, 1].ravel()
                yd, yi, md, _ = F.sample(vb, vbo, rng)
                X[k] = yd.reshape(len(rows), Nch, F.Nl)
                Y[k] = yi.reshape(len(rows), Nch, F.Nl)
                MU[k] = md.reshape(len(rows), Nch, F.Nl)
            fn = 'tau%05.1f.npz' % tau
            np.savez_compressed(os.path.join(outdir, fn), X=X, Y=Y, MU=MU, tau_ns=tau)
            manifest['files'].append(dict(file=fn, tau_ns=tau))
            print('  tau=%.1f ns  %s  %.0fs' % (tau, X.shape, time.time() - t0), flush=True)

    with open(os.path.join(outdir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print('\nwrote %s  (%d settings, %.0fs)' % (outdir, len(manifest['files']), time.time() - t0))


if __name__ == '__main__':
    main()

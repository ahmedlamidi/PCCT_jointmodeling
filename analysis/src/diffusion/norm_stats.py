"""Fixed normalisation, computed ONCE from the baseline arm and reused everywhere.

This must not be recomputed per dataset. If arm A is standardised by A's own
statistics and arm B by B's, the two models see different input scales and the
coverage comparison measures normalisation drift instead of forward-model
mismatch -- which is the entire experiment.

    python3 norm_stats.py                  # build from baseline3d, cache to JSON
    python3 norm_stats.py --show

Targets are standardised to EXACTLY zero mean / unit variance, so EDM's
sigma_data = 1.0 by construction (see edm.py).
"""
import argparse, json, os
import numpy as np
import sys; sys.path.insert(0, '..')
from paths import OUT

CACHE = os.path.join(OUT, 'norm_stats.json')
# The arm the statistics are DEFINED by. Never change this to a mismatched arm.
REF_ARM = 'baseline3d'


def _load_split(arm, split, nfile=None):
    with open(os.path.join(OUT, arm, 'manifest.json')) as f:
        man = json.load(f)
    files = man[split][:nfile] if nfile else man[split]
    return [os.path.join(OUT, arm, f) for f in files]


def build(arm=REF_ARM, nfile=3, nsamp=400_000, seed=0):
    """Per-channel mean/std of log1p(counts) and of the material line integrals."""
    rng = np.random.default_rng(seed)
    xs, ys, vs = [], [], []
    for fp in _load_split(arm, 'train', nfile):
        d = np.load(fp)
        X, Y, V = d['X'], d['Y'], d['V']
        nv, nr, nc, _ = X.shape
        n = nsamp // nfile
        iv = rng.integers(0, nv, n); ir = rng.integers(0, nr, n); ic = rng.integers(0, nc, n)
        xs.append(np.log1p(np.maximum(X[iv, ir, ic], 0)))
        ys.append(np.log1p(np.maximum(Y[iv, ir, ic], 0)))
        vs.append(V[iv, ir, ic])
    x = np.concatenate(xs); y = np.concatenate(ys); v = np.concatenate(vs)
    st = dict(
        arm=arm, transform='log1p for counts, identity for material paths',
        x_mean=x.mean(0).tolist(), x_std=(x.std(0) + 1e-6).tolist(),
        y_mean=y.mean(0).tolist(), y_std=(y.std(0) + 1e-6).tolist(),
        v_mean=v.mean(0).tolist(), v_std=(v.std(0) + 1e-6).tolist(),
        # FALLBACK ceiling for denorm_t, used only when the manifest has no
        # 'air_counts' (the generator now writes the exact value). This is a data
        # maximum: it equals the air count only if the arm contains air rays. A
        # narrow crop has none, and then it is merely the largest count seen.
        y_max_counts=np.expm1(y.max(0)).tolist(),
        v_max=v.max(0).tolist(),
    )
    with open(CACHE, 'w') as f:
        json.dump(st, f, indent=2)
    return st


def load():
    if not os.path.exists(CACHE):
        raise SystemExit('run: python3 norm_stats.py   (builds %s)' % CACHE)
    with open(CACHE) as f:
        return json.load(f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--arm', default=REF_ARM)
    a = ap.parse_args()
    st = load() if a.show else build(a.arm)
    print('normalisation defined by arm:', st['arm'])
    for k in ('x_mean', 'x_std', 'y_mean', 'y_std', 'v_mean', 'v_std', 'y_max_counts', 'v_max'):
        if k in st:
            print('  %-12s %s' % (k, np.round(st[k], 4)))
    print('\ncached at', CACHE)

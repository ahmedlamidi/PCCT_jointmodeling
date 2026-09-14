"""Patch loader for the diffusion arms.

Yields (cond, target) with the FIXED normalisation from norm_stats.py.

    target 'V'  material line integrals, 2 ch  -- the joint decomposition problem
    target 'Y'  ideal counts, 9 ch             -- Morovati's sinogram-correction task

Condition is always the distorted counts X (9 ch), log1p-standardised.
Patches are cut from the DETECTOR PLANE (row, channel) at one view, matching
Morovati's (u, v) patching.

Memory: one 4D phantom is ~770 MB decompressed, so ONE file is held at a time
and a call draws all its patches from that file. Drawing a random file per patch
thrashes the cache -- that was a real OOM here.
"""
import json, os
import numpy as np
import sys; sys.path.insert(0, '..')
from paths import OUT
import norm_stats


class DiffusionPatches:
    def __init__(self, arm, split='train', patch=16, target='V', seed=0,
                 reuse=64):
        """reuse: how many consecutive batches are drawn from one phantom file.

        A 4D phantom is ~770 MB decompressed. Drawing a random FILE per batch
        re-decompresses it on nearly every call and makes training I/O bound;
        holding one file for `reuse` batches removes that. Patch positions are
        still random, and the file is reselected often enough that a long run
        sees every phantom many times.
        """
        self.root = os.path.join(OUT, arm)
        with open(os.path.join(self.root, 'manifest.json')) as f:
            self.man = json.load(f)
        self.files = [os.path.join(self.root, f) for f in self.man[split]]
        if not self.files:
            raise ValueError('no files for split %r in %s' % (split, self.root))
        self.p = patch
        self.target = target
        st = norm_stats.load()
        self.xm = np.array(st['x_mean'], np.float32); self.xs = np.array(st['x_std'], np.float32)
        key = 'v' if target == 'V' else 'y'
        self.tm = np.array(st['%s_mean' % key], np.float32)
        self.ts = np.array(st['%s_std' % key], np.float32)
        # Physical ceiling. For counts: the unattenuated AIR count per bin, which
        # the generator computes from the forward model and writes to the
        # manifest -- exact, whether or not the crop contains air rays.
        # Fallbacks: the stats file's data maximum, then no upper clamp.
        if target == 'Y' and 'air_counts' in self.man:
            self.tmax = np.array(self.man['air_counts'], np.float32)
        else:
            mk = 'y_max_counts' if target == 'Y' else 'v_max'
            self.tmax = np.array(st[mk], np.float32) if mk in st else None
        self.cond_ch = len(self.xm)
        self.tgt_ch = len(self.tm)
        self._cache = (None, None)
        self.rng = np.random.default_rng(seed)
        self.reuse = max(1, int(reuse))
        self._left = 0
        self._cur = None

    def _load(self, i):
        if self._cache[0] != i:
            d = np.load(self.files[i])
            self._cache = (i, (d['X'], d[self.target]))
        return self._cache[1]

    def _norm_x(self, a):
        return (np.log1p(np.maximum(a, 0)) - self.xm) / self.xs

    def _norm_t(self, a):
        if self.target == 'V':
            return (a - self.tm) / self.ts
        return (np.log1p(np.maximum(a, 0)) - self.tm) / self.ts

    def denorm_t(self, a):
        """a: (..., C) normalised -> physical units (cm for V, counts for Y).

        Clipped to the physically possible range [0, ceiling]: no ray records
        fewer than 0 counts or more than the unattenuated air beam in that bin.
        The upper clip matters: samples return to counts through expm1, so one
        tail sample at +5 sigma in log space becomes ~1e6 counts and dominates
        an arithmetic posterior mean -- an untrained model scored RMSE 174,353
        counts against a true maximum of ~2,000. Both methods go through this
        same function, so the treatment is identical.
        """
        z = a * self.ts + self.tm
        if self.target == 'V':
            out = np.maximum(z, 0.0)
        else:
            out = np.maximum(np.expm1(np.minimum(z, 30.0)), 0.0)   # 30: no overflow
        if self.tmax is not None:
            out = np.minimum(out, self.tmax)
        return out

    def batch(self, n=32, rng=None):
        rng = rng or self.rng
        if self._left <= 0 or self._cur is None:
            self._cur = int(rng.integers(len(self.files)))
            self._left = self.reuse
        self._left -= 1
        X, T = self._load(self._cur)
        nv, nr, nc, _ = X.shape
        p = self.p
        xs, ts = [], []
        for _ in range(n):
            v = int(rng.integers(nv))
            i = int(rng.integers(0, nr - p + 1))
            j = int(rng.integers(0, nc - p + 1))
            xs.append(self._norm_x(X[v, i:i+p, j:j+p]).transpose(2, 0, 1))
            ts.append(self._norm_t(T[v, i:i+p, j:j+p]).transpose(2, 0, 1))
        return (np.stack(xs).astype(np.float32), np.stack(ts).astype(np.float32))

    def fixed_eval_set(self, n=256, seed=1234):
        """A held-out patch set: deterministic, and IDENTICAL for both methods
        (WGAN evaluate.py and diffusion coverage.py call this with the same n and
        seed) and across arms. Drawn evenly from EVERY phantom in the split -- it
        used to take all n patches from phantom 0, so the 'test set' was one
        phantom."""
        keep = (self._cur, self._left)
        nf = len(self.files)
        xs, ys = [], []
        for k in range(nf):
            m = n // nf + (1 if k < n % nf else 0)
            if m == 0:
                continue
            self._cur, self._left = k, self.reuse
            x, y = self.batch(m, np.random.default_rng(seed + k))
            xs.append(x); ys.append(y)
        self._cur, self._left = keep
        return np.concatenate(xs), np.concatenate(ys)

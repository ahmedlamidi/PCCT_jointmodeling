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


def _mem_limit():
    """Bytes this process may use: the smallest of node MemAvailable, the
    cgroup limit, and SLURM's --mem. A SLURM job is killed at its cgroup limit
    even when the node has far more free, so MemAvailable alone is not enough."""
    cands = []
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                cands.append(int(line.split()[1]) * 1024)
    except OSError:
        pass
    try:
        for line in open('/proc/self/cgroup'):
            path = line.strip().split(':', 2)[2]
            for f in ('/sys/fs/cgroup%s/memory.max' % path,
                      '/sys/fs/cgroup/memory%s/memory.limit_in_bytes' % path):
                if os.path.exists(f):
                    v = open(f).read().strip()
                    if v.isdigit() and int(v) < 1 << 60:
                        cands.append(int(v))
    except (OSError, IndexError):
        pass
    if os.environ.get('SLURM_MEM_PER_NODE', '').isdigit():
        cands.append(int(os.environ['SLURM_MEM_PER_NODE']) * 1024 ** 2)
    return min(cands) if cands else None


class DiffusionPatches:
    def __init__(self, arm, split='train', patch=16, target='V', seed=0,
                 reuse=64, preload=False):
        """reuse: how many consecutive batches are drawn from one phantom file.

        A 4D phantom is ~770 MB decompressed. Drawing a random FILE per batch
        re-decompresses it on nearly every call and makes training I/O bound;
        holding one file for `reuse` batches removes that. Patch positions are
        still random, and the file is reselected often enough that a long run
        sees every phantom many times.

        preload: load EVERY phantom of the split into RAM once, and draw each
        patch from a random phantom. Measured 2026-09-14: without it a WGAN
        iteration (6 batches) switched phantom every 10.7 iterations at 3.4 s
        per switch -- ~90% of its time was decompression, capping it at ~3 it/s.
        Costs ~0.77 GB per full-width phantom; refuses up front if the job's
        memory limit cannot hold it, instead of being OOM-killed mid-load.
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
        self._all = None
        if preload:
            self._preload()

    def _preload(self):
        import time
        m = self.man
        per = int(m['nview']) * int(m['nrow']) * int(m['nch']) * (self.cond_ch + self.tgt_ch) * 4
        need = per * len(self.files)
        avail = _mem_limit()
        if avail is not None and need > 0.8 * avail:
            raise MemoryError('--preload needs %.1f GB for %d phantoms but this job can use %.1f GB. '
                              'Drop --preload, or request more memory (--mem).'
                              % (need / 1e9, len(self.files), avail / 1e9))
        t = time.time()
        self._all = []
        for f in self.files:
            with np.load(f) as d:
                self._all.append((d['X'], d[self.target]))
        print('preloaded %d phantoms (%.1f GB) in %.0f s' % (len(self._all), need / 1e9, time.time() - t),
              flush=True)

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
        if self._all is not None:
            return self._batch_preloaded(n, rng)
        if self._left <= 0 or self._cur is None:
            self._cur = int(rng.integers(len(self.files)))
            self._left = self.reuse
        self._left -= 1
        X, T = self._load(self._cur)
        xs, ts = self._cut(X, T, n, rng)
        return (np.stack(xs).astype(np.float32), np.stack(ts).astype(np.float32))

    def _cut(self, X, T, n, rng):
        nv, nr, nc, _ = X.shape
        p = self.p
        xs, ts = [], []
        for _ in range(n):
            v = int(rng.integers(nv))
            i = int(rng.integers(0, nr - p + 1))
            j = int(rng.integers(0, nc - p + 1))
            xs.append(self._norm_x(X[v, i:i+p, j:j+p]).transpose(2, 0, 1))
            ts.append(self._norm_t(T[v, i:i+p, j:j+p]).transpose(2, 0, 1))
        return xs, ts

    def _batch_preloaded(self, n, rng):
        # every patch from an independently chosen phantom: no reloads, and
        # batches mix phantoms instead of coming 64 at a time from one
        xs, ts = [], []
        for _ in range(n):
            X, T = self._all[int(rng.integers(len(self._all)))]
            x, t = self._cut(X, T, 1, rng)
            xs += x; ts += t
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
            X, T = self._all[k] if self._all is not None else self._load(k)
            x, y = self._cut(X, T, m, np.random.default_rng(seed + k))
            xs.append(np.stack(x).astype(np.float32)); ys.append(np.stack(y).astype(np.float32))
        self._cur, self._left = keep
        return np.concatenate(xs), np.concatenate(ys)

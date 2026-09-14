"""Patch sampling from the saved sinograms.

generate_baseline_dataset.py stores sinograms, not patches, because 1.8M
16x16x9 patches would be ~17 GB while the sinograms are ~24 MB per phantom.
Patches are cut here, on the fly.

Patch axes are (channel, view); the bin axis is the channel dimension of the
network, like RGB in a photo.

    from patches import PatchSet
    ps = PatchSet('baseline_ds', split='train')
    x, y = ps.sample(batch=64, rng=rng)      # (64, Nl, 16, 16) each

    # deterministic full sweep, e.g. for validation
    for x, y in ps.iter_grid(batch=64): ...

A torch Dataset is provided only if torch imports; nothing here requires it.

NORMALISATION. Counts span ~1e0 to ~1e6 across bins and rays, so feed a network
log1p or per-bin standardised values, not raw counts. `stats()` returns per-bin
mean and sd over a sample so you can fix a normalisation once and reuse it for
every operator setting -- do NOT recompute it per dataset, or arms become
incomparable.
"""
import json, os
import numpy as np
from paths import OUT


class PatchSet:
    def __init__(self, name, split='train', patch=None, stride=None,
                 target='Y', input_key='X', cache=None):
        """target 'Y' = ideal counts (label). 'MU_D' = noise-free distorted mean.

        cache: how many decompressed phantoms to hold. A 4D phantom is ~770 MB
        (X and Y at 180x32x1854x9 float32), so the default is 1 for 4D data and
        3 for 2D. Raising it on a 16 GB box will OOM.
        """
        self.root = os.path.join(OUT, name)
        with open(os.path.join(self.root, 'manifest.json')) as f:
            self.man = json.load(f)
        self.files = [os.path.join(self.root, f) for f in self.man[split]]
        if not self.files:
            raise ValueError('no files for split %r in %s' % (split, self.root))
        self.p = patch or self.man['patch']
        self.s = stride or self.man['stride']
        self.target, self.input_key = target, input_key
        self.Nl = self.man['bins']
        self._cache = {}
        # 4D phantoms are ~770 MB decompressed; hold one unless told otherwise
        probe = np.load(self.files[0])[self.input_key].ndim if cache is None else 0
        self.cache_n = cache if cache is not None else (1 if probe == 4 else 3)

    # ------------------------------------------------------------------
    def _load(self, i):
        if i not in self._cache:
            d = np.load(self.files[i])
            self._cache[i] = (d[self.input_key], d[self.target])
            while len(self._cache) > self.cache_n:        # keep memory bounded
                self._cache.pop(next(iter(self._cache)))
        return self._cache[i]

    def _cut(self, X, Y, i, j, view=None):
        """Cut one patch -> (Nl, p, p) for both arrays.

        3D array (nview, nch, Nl):        patch over (view, channel).
        4D array (nview, nrow, nch, Nl):  patch over (row, channel) at one view,
                                          i.e. the detector plane (u, v), which
                                          is what Morovati patch.
        """
        p = self.p
        if X.ndim == 4:
            x = X[view, i:i + p, j:j + p].transpose(2, 0, 1)
            y = Y[view, i:i + p, j:j + p].transpose(2, 0, 1)
        else:
            x = X[i:i + p, j:j + p].transpose(2, 0, 1)
            y = Y[i:i + p, j:j + p].transpose(2, 0, 1)
        return x, y

    # ------------------------------------------------------------------
    def sample(self, batch=64, rng=None):
        """Random patches across random phantoms."""
        rng = np.random.default_rng() if rng is None else rng
        # ONE phantom per call. Drawing a random file per patch evicted the
        # cache on nearly every draw -- that was the OOM and the slowness.
        X, Y = self._load(int(rng.integers(len(self.files))))
        four = X.ndim == 4
        ni, nj = (X.shape[1], X.shape[2]) if four else (X.shape[0], X.shape[1])
        xs, ys = [], []
        for _ in range(batch):
            view = int(rng.integers(X.shape[0])) if four else None
            i = int(rng.integers(0, ni - self.p + 1))
            j = int(rng.integers(0, nj - self.p + 1))
            x, y = self._cut(X, Y, i, j, view)
            xs.append(x); ys.append(y)
        return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)

    def iter_grid(self, batch=64, views=None):
        """Deterministic sweep over every patch position, for validation.

        `views` limits how many projections are swept per phantom; the full
        sweep over 4D data is ~1.9M patches.
        """
        xs, ys = [], []
        for k in range(len(self.files)):
            X, Y = self._load(k)
            four = X.ndim == 4
            vs = range(X.shape[0]) if four else [None]
            if four and views:
                vs = range(0, X.shape[0], max(1, X.shape[0] // views))
            for view in vs:
                ni, nj = (X.shape[1], X.shape[2]) if four else (X.shape[0], X.shape[1])
                for i in range(0, ni - self.p + 1, self.s):
                    for j in range(0, nj - self.p + 1, self.s):
                        x, y = self._cut(X, Y, i, j, view)
                        xs.append(x); ys.append(y)
                        if len(xs) == batch:
                            yield (np.stack(xs).astype(np.float32),
                                   np.stack(ys).astype(np.float32))
                            xs, ys = [], []
        if xs:
            yield np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)

    # ------------------------------------------------------------------
    def stats(self, n=4000, rng=None):
        """Per-bin mean and sd of log1p(counts). Fix these ONCE and reuse."""
        x, y = self.sample(n, rng)
        lx, ly = np.log1p(np.maximum(x, 0)), np.log1p(np.maximum(y, 0))
        return dict(x_mean=lx.mean((0, 2, 3)), x_std=lx.std((0, 2, 3)) + 1e-8,
                    y_mean=ly.mean((0, 2, 3)), y_std=ly.std((0, 2, 3)) + 1e-8)


def torch_dataset(patchset, length=100000, seed=0):
    """Optional torch wrapper. Raises ImportError if torch is absent."""
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __init__(self):
            self.ps = patchset
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return length

        def __getitem__(self, idx):
            x, y = self.ps.sample(1, self.rng)
            return torch.from_numpy(x[0]), torch.from_numpy(y[0])

    return _DS()

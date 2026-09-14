"""Blend overlapping patch predictions back into a full detector plane.

Overlapping the patches (16x16 at stride 8) is only half the job. If you drop
each patch into place and let the last one win, you get visible seams. If you
plain-average them, the seams are fainter but still there, because a pixel near
a patch BORDER was predicted with less surrounding context than one in the
middle, so its prediction is worse.

Fix: weight each pixel by how far it is from its patch's edge, using a raised
cosine (Hann) window, and divide by the total weight. Centre pixels dominate,
edge pixels contribute little, and the weights sum to 1 everywhere.

    acc = Stitcher((nrow, nch), n_ch=2, patch=16, stride=8)
    for (i, j), pred in predictions:      # pred: (n_ch, 16, 16)
        acc.add(i, j, pred)
    full = acc.result()                   # (n_ch, nrow, nch)

`taper=False` gives the plain average, for comparison.
"""
import numpy as np


def hann2d(p):
    """2-D raised cosine, never exactly zero so every pixel keeps some weight."""
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(p) + 0.5) / p)
    w = np.maximum(w, 1e-3)
    return np.outer(w, w)


class Stitcher:
    def __init__(self, shape, n_ch, patch=16, stride=8, taper=True):
        self.p, self.s = patch, stride
        self.acc = np.zeros((n_ch,) + tuple(shape), np.float64)
        self.wsum = np.zeros(tuple(shape), np.float64)
        self.w = hann2d(patch) if taper else np.ones((patch, patch))

    def add(self, i, j, pred):
        """pred: (n_ch, patch, patch) placed with its top-left at (i, j)."""
        p = self.p
        self.acc[:, i:i+p, j:j+p] += pred * self.w
        self.wsum[i:i+p, j:j+p] += self.w

    def result(self):
        if (self.wsum <= 0).any():
            raise ValueError('%d pixels were never covered by a patch; check the '
                             'stride and the plane size' % int((self.wsum <= 0).sum()))
        return (self.acc / self.wsum).astype(np.float32)


def positions(shape, patch=16, stride=8):
    """Top-left corners covering the plane, with the last row/column pushed in
    so the far edge is covered even when (n - patch) is not a multiple of stride."""
    out = []
    for i in _starts(shape[0], patch, stride):
        for j in _starts(shape[1], patch, stride):
            out.append((i, j))
    return out


def _starts(n, patch, stride):
    if n < patch:
        raise ValueError('plane dimension %d smaller than patch %d' % (n, patch))
    s = list(range(0, n - patch + 1, stride))
    if s[-1] != n - patch:
        s.append(n - patch)
    return s

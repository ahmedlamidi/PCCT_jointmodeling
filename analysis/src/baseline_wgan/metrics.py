"""PSNR / SSIM / RMSE -- the metrics Morovati et al. report. Pure numpy.

Kept separate from evaluate.py so they can be tested without torch installed.
"""
import numpy as np


def psnr(a, b, peak=None):
    mse = ((a - b) ** 2).mean()
    peak = peak if peak is not None else max(b.max(), 1e-12)
    return 10 * np.log10(peak ** 2 / max(mse, 1e-20))


def ssim(a, b, C1=1e-4, C2=9e-4):
    """Global SSIM on normalised patches. Windowed SSIM needs scipy; this is the
    single-window form and is enough for a relative comparison between methods."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))

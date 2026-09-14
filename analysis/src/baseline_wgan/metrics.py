"""PSNR / SSIM / RMSE -- the metrics Morovati et al. report. numpy (+ scipy for
ssim_windowed).

Kept separate from evaluate.py so they can be tested without torch installed.

ssim() is ONE global window over everything passed in. Called on (N, 9) counts it
reads ~0.9999 for any sensible prediction, because the spread across bins and rays
(hundreds of counts) dwarfs the error. ssim_windowed() is the standard local form.
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


def ssim_windowed(a, b, data_range, sigma=1.5, truncate=3.5, K1=0.01, K2=0.03):
    """Local-window SSIM, Wang et al. 2004 (doi 10.1109/TIP.2003.819861): Gaussian
    window sigma=1.5 (11x11 at truncate=3.5), K1=0.01, K2=0.03, population
    covariance. These are the settings of skimage.metrics.structural_similarity with
    gaussian_weights=True, use_sample_covariance=False, which this reproduces.

    a, b: (..., C, H, W). SSIM is taken over the last two axes only, so each
    channel (energy bin) of each patch is scored as its own image.
    data_range: scalar or (C,) -- L in C1=(K1 L)^2, C2=(K2 L)^2, per channel.
    Returns (..., C): mean local SSIM over the window positions that lie wholly
    inside the image (a border of r=(win-1)/2 is dropped, as skimage does).
    """
    from scipy.ndimage import gaussian_filter
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    r = int(truncate * sigma + 0.5)
    if min(a.shape[-2:]) <= 2 * r:
        raise ValueError('image %s too small for a %dx%d window' % (a.shape[-2:], 2*r+1, 2*r+1))
    L = np.asarray(data_range, np.float64)[..., None, None]
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    sig = (0,) * (a.ndim - 2) + (sigma, sigma)          # sigma 0 = no smoothing on that axis
    f = lambda z: gaussian_filter(z, sig, truncate=truncate)
    ua, ub = f(a), f(b)
    va, vb, cab = f(a * a) - ua ** 2, f(b * b) - ub ** 2, f(a * b) - ua * ub
    S = ((2 * ua * ub + C1) * (2 * cab + C2)) / ((ua ** 2 + ub ** 2 + C1) * (va + vb + C2))
    return S[..., r:-r, r:-r].mean(axis=(-2, -1))

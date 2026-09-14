"""Second domain of the dual-domain pipeline: TV-L1 then guided filtering.

Morovati et al. 2025: "TV-L1 denoising method optimized with a primal-dual
algorithm", followed by guided filtering that uses a virtual "integrating" bin --
the sum over all energy channels -- as the guidance image. Using the summed bin
as the guide is the point: it has far better SNR than any single energy bin, so
edges are taken from it while each bin keeps its own contrast.

Pure numpy, so it runs and is testable without torch.

    y = tv_l1(x, lam=0.15)                 # per energy bin
    z = guided_filter(y, guide=y.sum(0))   # edges from the summed bin
"""
import numpy as np


def tv_l1(img, lam=0.15, n_iter=100, tau=0.25, sigma=0.25):
    """TV-L1 denoising by the Chambolle-Pock primal-dual algorithm.

    img: (H, W) or (C, H, W) -- each channel is denoised independently.
    lam: larger = more fidelity to the input, less smoothing.
    """
    single = img.ndim == 2
    x = (img[None] if single else img).astype(np.float64)
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        f = x[c]
        u = f.copy(); u_bar = f.copy()
        p = np.zeros((2,) + f.shape)
        for _ in range(n_iter):
            gx, gy = _grad(u_bar)
            p[0] += sigma * gx
            p[1] += sigma * gy
            nrm = np.maximum(1.0, np.sqrt(p[0] ** 2 + p[1] ** 2))
            p /= nrm
            u_old = u
            v = u + tau * _div(p)
            # prox of tau*lam*|u - f|_1  (soft threshold toward f)
            d = v - f
            u = f + np.sign(d) * np.maximum(np.abs(d) - tau * lam, 0.0)
            u_bar = 2 * u - u_old
        out[c] = u
    return out[0] if single else out


def _grad(u):
    gx = np.zeros_like(u); gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def _div(p):
    px, py = p[0], p[1]
    dx = np.zeros_like(px); dy = np.zeros_like(py)
    dx[:, 0] = px[:, 0]; dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]; dx[:, -1] = -px[:, -2]
    dy[0, :] = py[0, :]; dy[1:-1, :] = py[1:-1, :] - py[:-2, :]; dy[-1, :] = -py[-2, :]
    return dx + dy


def _box(a, r):
    """Box filter of radius r via a summed-area table, edge-normalised."""
    k = 2 * r + 1
    c = np.cumsum(np.cumsum(np.pad(a, ((1, 0), (1, 0))), 0), 1)
    H, W = a.shape
    i0 = np.clip(np.arange(H) - r, 0, H); i1 = np.clip(np.arange(H) + r + 1, 0, H)
    j0 = np.clip(np.arange(W) - r, 0, W); j1 = np.clip(np.arange(W) + r + 1, 0, W)
    S = (c[np.ix_(i1, j1)] - c[np.ix_(i0, j1)] - c[np.ix_(i1, j0)] + c[np.ix_(i0, j0)])
    n = np.outer(i1 - i0, j1 - j0)
    return S / n


def guided_filter(img, guide, r=4, eps=1e-3):
    """He et al. guided filter. img: (H,W) or (C,H,W); guide: (H,W).

    The guide supplies the edges, the input supplies the values -- which is why
    the summed 'integrating' bin works: it is the same anatomy at much better SNR.
    """
    g = guide.astype(np.float64)
    mg = _box(g, r); mgg = _box(g * g, r)
    vg = mgg - mg * mg
    single = img.ndim == 2
    X = img[None] if single else img
    out = np.empty_like(X, dtype=np.float64)
    for c in range(X.shape[0]):
        p = X[c].astype(np.float64)
        mp = _box(p, r)
        cov = _box(g * p, r) - mg * mp
        a = cov / (vg + eps)
        b = mp - a * mg
        out[c] = _box(a, r) * g + _box(b, r)
    return out[0] if single else out


def image_domain(bins, tv_lam=0.15, tv_iter=100, gf_r=4, gf_eps=1e-3):
    """Full second stage: TV-L1 per bin, then guided filtering on the summed bin."""
    y = tv_l1(bins, lam=tv_lam, n_iter=tv_iter)
    return guided_filter(y, guide=y.sum(0), r=gf_r, eps=gf_eps)

"""Spatially correlated noise via 3x3 stamping, with a tunable sharing strength rho.

WHY THIS EXISTS
---------------
pipeline.Forward collapses PcTK's 36x36 neighbourhood covariance to a single
pixel by summing the nine diagonal blocks:

    C4 = sum_p  Cw[4p:4p+4, 4p:4p+4]

That keeps SPECTRAL correlation (between energy bins in one pixel) but throws
away SPATIAL correlation (between neighbouring pixels). This module keeps both,
by doing what script_workflow_PcTK.m does: draw the full 36-vector for each
source pixel and stamp it into the 3x3 neighbourhood, accumulating overlaps.

Pixel j's counts are then the sum of contributions from the 9 sources around it.
Neighbouring pixels share sources, which is where the spatial correlation comes from.

THE rho KNOB
------------
    C(rho) = rho * C  +  (1 - rho) * BlockDiag(C)

rho = 1  full PcTK sharing
rho = 0  no spatial correlation (each pixel independent) -- hardware anti-coincidence
Both endpoints are PSD, and so is every rho in [0,1], because C(rho) is a convex
combination of two PSD matrices.

IMPORTANT LIMITATION: rho scales the noise correlation ONLY. The diagonal, and
therefore the mean, is untouched. Real anti-coincidence also recombines shared
charge and so changes the mean. The mean-level analogue is the existing
PIXELS='center' toggle in pctk_compare.CFG. Keeping them separate is deliberate --
it lets you vary noise correlation and bias independently.

INDEXING (matches PcTK)
-----------------------
36-vector index = bin + 4*p, with p = ix + 3*iy, ix,iy in {0,1,2}.
Offset from the source pixel is (ix-1, iy-1); p=4 is the centre.

APPROXIMATION
-------------
The matrix square root L is precomputed on a (brain, bone) grid and interpolated,
rather than factorised per pixel. L_interp @ L_interp.T is close to, but not
exactly, the interpolated covariance. It is always a valid covariance. Tighten by
refining the grid if you need it exact.
"""
import numpy as np
import h5py
from paths import PCTK


NCOV_FILE = (PCTK + '/1_inputdata/dat_nCovw_ver3.2_dpix_225_dz_1600_r0_24_esig_2.0_'
                    'Eth_20.0_50.0_65.0_80.0_keV.mat')


def block_diag_part(C):
    """Zero every 4x4 block off the pixel diagonal. C: (...,36,36)."""
    out = np.zeros_like(C)
    for p in range(9):
        s = slice(4 * p, 4 * p + 4)
        out[..., s, s] = C[..., s, s]
    return out


class SpatialNoise:
    """Correlated 3x3 count generation for one detector row-block.

        sn = SpatialNoise(F, rho=1.0)
        y  = sn.sample_view(vb2d, vbo2d, rng)      # (Nl, Nch, Nrow)

    vb2d, vbo2d are (Nch, Nrow) path lengths in cm for a single view.
    """

    def __init__(self, F, rho=1.0, grid_brain=None, grid_bone=None, verbose=True):
        self.F, self.rho = F, float(rho)
        self.Nl = F.Nl
        if self.Nl != 4:
            raise ValueError('SpatialNoise assumes PcTK\'s 4 windows (36 = 9 pixels x 4 bins); '
                             'got Nl=%d. Rebin in the E domain first.' % self.Nl)
        with h5py.File(NCOV_FILE, 'r') as h:
            self.C36 = np.array(h['m3_nCov3x3w'])          # (170, 36, 36)

        self.gb = (np.concatenate([np.arange(0, 8, 1.0), np.arange(8, 40, 2.0)])
                   if grid_brain is None else np.asarray(grid_brain, float))
        self.gc = (np.linspace(0, 6, 13) if grid_bone is None
                   else np.asarray(grid_bone, float))
        self._build(verbose)

    # ------------------------------------------------------------------
    def _cov(self, a, b):
        """36x36 covariance for one operating point, with rho applied."""
        sp = self.F.spec(np.array([a]), np.array([b]))[:, 0]        # (170,)
        C = self.F.N0 * np.einsum('e,eij->ij', sp, self.C36)
        C = 0.5 * (C + C.T)
        if self.rho != 1.0:
            C = self.rho * C + (1.0 - self.rho) * block_diag_part(C)
        return C

    def _build(self, verbose):
        nb, nc = len(self.gb), len(self.gc)
        self.MU = np.zeros((nb, nc, 36))
        self.L = np.zeros((nb, nc, 36, 36))
        for i, a in enumerate(self.gb):
            for j, b in enumerate(self.gc):
                C = self._cov(a, b)
                self.MU[i, j] = np.diag(C)                 # PcTK: mean = diag(cov)
                w, V = np.linalg.eigh(C)
                self.L[i, j] = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
            if verbose:
                print('  spatial-noise grid  brain %5.1f cm' % a, flush=True)

    # ------------------------------------------------------------------
    def _interp(self, a, b):
        """Bilinear lookup of (mean, L) for flat arrays of path lengths."""
        x = np.clip(np.interp(a, self.gb, np.arange(len(self.gb))), 0, len(self.gb) - 1 - 1e-9)
        y = np.clip(np.interp(b, self.gc, np.arange(len(self.gc))), 0, len(self.gc) - 1 - 1e-9)
        i0 = x.astype(int); j0 = y.astype(int)
        fx = (x - i0)[:, None]; fy = (y - j0)[:, None]
        M = self.MU
        mu = ((M[i0, j0] * (1 - fx) + M[i0 + 1, j0] * fx) * (1 - fy) +
              (M[i0, j0 + 1] * (1 - fx) + M[i0 + 1, j0 + 1] * fx) * fy)
        L = self.L; fx2 = fx[..., None]; fy2 = fy[..., None]
        Lc = ((L[i0, j0] * (1 - fx2) + L[i0 + 1, j0] * fx2) * (1 - fy2) +
              (L[i0, j0 + 1] * (1 - fx2) + L[i0 + 1, j0 + 1] * fx2) * fy2)
        return mu, Lc

    # ------------------------------------------------------------------
    def sample_view(self, vb2d, vbo2d, rng, z=None, noise_free=False):
        """One view. vb2d, vbo2d: (Nch, Nrow) cm. Returns (Nl, Nch, Nrow).

        Pass z of shape (Nch*Nrow, 36) to reuse a draw across operator settings.
        """
        vb2d = np.asarray(vb2d, float); vbo2d = np.asarray(vbo2d, float)
        Nch, Nrow = vb2d.shape
        mu, L = self._interp(vb2d.ravel(), vbo2d.ravel())          # (B,36), (B,36,36)
        if noise_free:
            d = mu
        else:
            if z is None:
                z = rng.standard_normal((mu.shape[0], 36))
            d = mu + np.einsum('bij,bj->bi', L, z)
        # 36-vector index = bin + Nl*p with p = ix + 3*iy, so BIN VARIES FASTEST.
        # C-order reshape must therefore be (iy, ix, bin), not (bin, ix, iy).
        d = d.reshape(Nch, Nrow, 3, 3, self.Nl)

        pad = np.zeros((self.Nl, Nch + 2, Nrow + 2))
        for p in range(9):
            ix, iy = p % 3, p // 3
            pad[:, ix:ix + Nch, iy:iy + Nrow] += d[:, :, iy, ix, :].transpose(2, 0, 1)
        return np.maximum(pad[:, 1:-1, 1:-1], 0.0)

    # ------------------------------------------------------------------
    def measured_correlation(self, a, b, n=20000, nrow=5, rng=None):
        """Empirical adjacent-channel correlation at one operating point.

        Sanity check against Gate 2: rho=1 should land near 0.06-0.10,
        rho=0 near 0.

        nrow must be >= 3. With a single row the pixels of interest receive
        only 3 of their 9 neighbour contributions -- the row-direction
        neighbours fall into the padding and are trimmed -- which deflates both
        the mean and the correlation. The middle row is the one read out.
        """
        if nrow < 3:
            raise ValueError('nrow must be >= 3 so row-neighbour stamping is present')
        rng = np.random.default_rng(0) if rng is None else rng
        vb = np.full((n, nrow), a); vbo = np.full((n, nrow), b)
        y = self.sample_view(vb, vbo, rng)[:, :, nrow // 2].sum(0)   # counts per channel
        y = y[10:-10]
        return float(np.corrcoef(y[:-1], y[1:])[0, 1])

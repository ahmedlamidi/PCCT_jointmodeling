"""Neighbourhood covariance at ARBITRARY energy thresholds.

WHY
---
PcTK ships nCov3x3w only for its own 20/50/65/80 keV thresholds, and
pipeline.Forward hardcodes 4 bins:

    C4 = sum(Cw[:, 4*p:4*p+4, 4*p:4*p+4] for p in range(9))

To match another paper's binning (Morovati uses 9 bins, 20-109 keV) or your own
detector (2 bins at 30 keV) the covariance has to be rebinned from the full
energy-resolved nCovE, which is the 4 GB array. The cached diagonal is not
enough -- covariance needs the off-diagonal terms.

HOW
---
nCovE is indexed by (9 pixels x 191 output-energy bins) = 1719 on each axis,
energy fastest:  index = 191*p + eo.

Build an indicator A of shape (1719, 9*Nl):

    A[191*p + eo, Nl*p + l] = 1   if eo falls in window l

then the rebinned covariance for incident energy E is just

    C_w(E) = A.T @ C_E(E) @ A

Output layout matches PcTK: index = Nl*p + l, p = ix + 3*iy, so p=4 is centre.

COST
----
Streams the 4 GB file once, roughly 7-10 minutes. Cached to
cache/ncovw_<thresholds>.npz, so you pay it once per threshold set.

VALIDATION
----------
--validate rebins at PcTK's own 20/50/65/80 and compares against the shipped
nCov3x3w. Should agree to ~1e-12. Run it once before trusting a new binning.

    python3 rebin_cov.py --eth 20,30,40,50,60,70,80,90,100
    python3 rebin_cov.py --validate
"""
import argparse, os, time
import numpy as np
import h5py
from paths import PCTK, CACHE

NCOVE = PCTK + '/5_refdata/dat_nCovE_ver3.2_dpix_225_dz_1600_r0_24_esig_2.0.mat'
NCOVW = (PCTK + '/1_inputdata/dat_nCovw_ver3.2_dpix_225_dz_1600_r0_24_esig_2.0_'
                'Eth_20.0_50.0_65.0_80.0_keV.mat')


def tag(eth):
    return 'ncovw_' + '_'.join('%g' % t for t in eth)


def indicator(vEo, eth):
    """(1719, 9*Nl) 0/1 matrix mapping energy-resolved index -> (pixel, window)."""
    eth = np.asarray(eth, float)
    Nl = len(eth)
    nE = len(vEo)                                    # 191
    win = np.digitize(vEo, eth) - 1                  # -1 below the lowest threshold
    A = np.zeros((9 * nE, 9 * Nl), np.float64)
    for p in range(9):
        for e in range(nE):
            l = win[e]
            if l >= 0:
                A[nE * p + e, Nl * p + l] = 1.0
    return A


def build(eth, out=None, verbose=True):
    """Rebin nCovE to the given thresholds. Returns (C, vE1) with C (170, 9Nl, 9Nl)."""
    eth = [float(x) for x in eth]
    Nl = len(eth)
    with h5py.File(NCOVE, 'r') as h:
        vEo = np.array(h['v_Eo']).ravel()
        vE1 = np.array(h['v_E1']).ravel()
        A = indicator(vEo, eth)
        D = h['m3_nCov3x3E']
        n = D.shape[0]
        C = np.zeros((n, 9 * Nl, 9 * Nl))
        t0 = time.time()
        for i in range(n):
            S = np.array(D[i])                       # (1719, 1719)
            C[i] = A.T @ S @ A
            if verbose and i % 20 == 0:
                print('  slice %3d/%d   %.0fs' % (i, n, time.time() - t0), flush=True)
    if out is None:
        out = os.path.join(CACHE, tag(eth) + '.npz')
    np.savez_compressed(out, C=C, eth=np.asarray(eth), vE1=vE1)
    if verbose:
        print('wrote %s   C%s   %.0fs' % (out, C.shape, time.time() - t0))
    return C, vE1


def load(eth, build_if_missing=True):
    """Cached lookup. Returns C (170, 9*Nl, 9*Nl)."""
    p = os.path.join(CACHE, tag([float(x) for x in eth]) + '.npz')
    if os.path.exists(p):
        return np.load(p)['C']
    if not build_if_missing:
        raise FileNotFoundError(p)
    return build(eth)[0]


def per_pixel(C, Nl):
    """Collapse to the single-pixel covariance: sum of the 9 diagonal blocks.
    This is what pipeline.Forward uses (spectral correlation, no spatial)."""
    return sum(C[:, Nl * p:Nl * p + Nl, Nl * p:Nl * p + Nl] for p in range(9))


def validate():
    """Rebin at PcTK's own thresholds and compare against the shipped table."""
    eth = [20., 50., 65., 80.]
    print('rebinning at PcTK thresholds %s ...' % eth)
    C, _ = build(eth, out=os.path.join(CACHE, tag(eth) + '.npz'))
    with h5py.File(NCOVW, 'r') as h:
        ref = np.array(h['m3_nCov3x3w'])             # (170, 36, 36)
    err = np.abs(C - ref).max()
    print('\nmax |rebinned - shipped nCov3x3w| = %.3e   (scale %.3e)' % (err, np.abs(ref).max()))
    print('PASS' if err < 1e-9 else 'FAIL -- do not trust other binnings')
    return err


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--eth', default='20,30,40,50,60,70,80,90,100',
                    help='comma-separated thresholds in keV')
    ap.add_argument('--validate', action='store_true')
    a = ap.parse_args()
    if a.validate:
        validate()
    else:
        build([float(x) for x in a.eth.split(',')])

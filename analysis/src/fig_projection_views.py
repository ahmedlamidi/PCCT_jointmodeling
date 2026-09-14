"""Projection views at one angle in a chosen energy window -- reproduction of
the reference Figure 6(a).

Four panels, same labels as the paper:
    Before Detection (Sino)              perfect detector
    After Charge Splitting (CS)          PcTK, noise free
    Charge Splitting with Poisson Noise  CS + noise
    After Pulse Pileup (PU)              + pile-up, + noise

This is the DETECTOR PLANE (rows x channels) at a single view, not a sinogram
and not a reconstruction. The red dot marks the open-beam pixel whose spectrum
fig_open_beam_spectrum.py plots.
"""
import argparse, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '.')
from paths import FIGS, OUT
import pctk_compare as P
from pipeline import Forward
from projector import Geometry
from phantom import head3d

ap = argparse.ArgumentParser()
ap.add_argument('--nrow', type=int, default=300, help='detector rows = axial slices')
ap.add_argument('--npix', type=int, default=256)
ap.add_argument('--view', type=int, default=1, help='view index (nview=4, so 1 = 90 deg)')
ap.add_argument('--bin', type=int, default=5, help='1-based energy window; 5 = 60-70 keV')
ap.add_argument('--tau', type=float, default=30.0)
ap.add_argument('--maxrays', type=int, default=250_000)
ap.add_argument('--fov', type=float, default=60.0,
                help='reconstruction FOV in mm. The reference object transmits ~64%% at '
                     '60 keV; a 21 cm head transmits 0.7%% and renders solid black, so the '
                     'phantom is scaled to an extremity-sized object to match.')
A = ap.parse_args()

ETH = [20., 30., 40., 50., 60., 70., 80., 90., 100.]
LBL = ['%g-%s' % (ETH[i], ('%g' % ETH[i+1]) if i < 8 else 'inf') for i in range(9)]
bi = A.bin - 1
t0 = time.time()

print('building 3D phantom (%d slices)...' % A.nrow, flush=True)
brain, bone = head3d(A.npix, A.nrow)
g = Geometry(nview=4, nch=1854, npix=A.npix, fov_mm=A.fov)
SB = np.empty((A.nrow, g.nch)); SBO = np.empty_like(SB)
for r in range(A.nrow):
    SB[r] = g.forward(brain[r])[:, A.view]
    SBO[r] = g.forward(bone[r])[:, A.view]
print('projected  %.0fs   soft<=%.1f cm bone<=%.1f cm' % (time.time()-t0, SB.max(), SBO.max()), flush=True)

cfg = dict(P.CFG); cfg['ETH'] = ETH
F0 = Forward(cfg=cfg, tau_ns=0.0, verbose=False)
import os
gcache = OUT + '/pugrid_tau%g_bins9.npz' % A.tau
Fp = Forward(cfg=cfg, tau_ns=0.0, verbose=False)
z = np.load(gcache)
Fp.grid = dict(brain=z['brain'], bone=z['bone'], mean=z['mean'], cov=z['cov'])
print('loaded cached pile-up grid', flush=True)

rng = np.random.default_rng(0)
n = A.nrow * g.nch
MI = np.empty(n); MD = np.empty(n); MP = np.empty(n)
vb, vbo = SB.ravel(), SBO.ravel()
step = A.maxrays
for s in range(0, n, step):
    e = min(s + step, n)
    md, _, mi = F0.mean_cov(vb[s:e], vbo[s:e])
    mp, _, _ = Fp.mean_cov(vb[s:e], vbo[s:e])
    MI[s:e] = mi[:, bi]; MD[s:e] = md[:, bi]; MP[s:e] = mp[:, bi]
print('counts  %.0fs' % (time.time()-t0), flush=True)

sino = MI.reshape(A.nrow, g.nch)
cs = MD.reshape(A.nrow, g.nch)
cspn = rng.poisson(np.maximum(cs, 0)).astype(float)
pu = rng.poisson(np.maximum(MP.reshape(A.nrow, g.nch), 0)).astype(float)

# crop to the object support plus a margin of open beam, for a sensible aspect
prof = sino.min(0)
obj = np.where(prof < prof.max() * 0.98)[0]
c0, c1 = max(obj[0] - 120, 0), min(obj[-1] + 120, g.nch)
panels = [('Before Detection (Sino)', sino), ('After Charge Splitting (CS)', cs),
          ('Charge Splitting with Poisson Noise (CSPN)', cspn), ('After Pulse Pileup (PU)', pu)]

plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(2, 2, figsize=(13, 7.4))
for k, (t, img) in enumerate(panels):
    a = ax[k // 2, k % 2]
    m = img[:, c0:c1]
    lo = np.percentile(m, 1.0); hi = np.percentile(m, 99.5)
    im = a.imshow(m, cmap='gray', aspect='auto', vmin=lo, vmax=hi)
    a.plot(0.03 * (c1 - c0), 0.06 * A.nrow, 'o', color='red', ms=6)
    a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=10, fontweight='bold')
    cb = plt.colorbar(im, ax=a, fraction=.030, pad=.02); cb.set_label('Counts', fontsize=9)
fig.suptitle('Projection views at %d° in the %s keV energy window'
             % (A.view * 90, LBL[bi]), fontsize=12)
fig.tight_layout()
fig.savefig(FIGS + '/physics_projection_views.png', dpi=140, bbox_inches='tight')

print('\n%-42s %10s %10s' % ('panel', 'open beam', 'centre'))
for t, img in panels:
    print('%-42s %10.0f %10.0f' % (t, img[int(.06*A.nrow), c0+10], img[A.nrow//2, g.nch//2]))
print('\nwrote figures/physics_projection_views.png  (%.0fs)' % (time.time()-t0))

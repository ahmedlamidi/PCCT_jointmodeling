"""Per-bin cascade, in the style of a before / truth / error comparison.

One row per energy bin. Columns:
    ideal            perfect detector
    +sharing+escape  PcTK, no pile-up
    +pile-up         full cascade
    relative error   (full - ideal)/ideal, %

Reuses the cached 9-bin pile-up grid if present, so this is minutes not hours.
"""
import argparse, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from paths import FIGS, OUT
import pctk_compare as P
from pipeline import Forward
from projector import Geometry
from phantom import head

ap = argparse.ArgumentParser()
ap.add_argument('--nview', type=int, default=360)
ap.add_argument('--npix', type=int, default=384)
ap.add_argument('--tau', type=float, default=30.0)
ap.add_argument('--cap', type=float, default=100.0)
ap.add_argument('--maxrays', type=int, default=250_000)
A = ap.parse_args()

ETH = [20., 30., 40., 50., 60., 70., 80., 90., 100.]
LBL = ['%g-%s' % (ETH[i], ('%g' % ETH[i+1]) if i < 8 else 'inf') for i in range(9)]
t0 = time.time()

brain, bone = head(A.npix)
g = Geometry(nview=A.nview, npix=A.npix)
sb, sbo = g.forward(brain), g.forward(bone)
print('projected  %.0fs' % (time.time()-t0), flush=True)

cfg = dict(P.CFG); cfg['ETH'] = ETH
F0 = Forward(cfg=cfg, tau_ns=0.0, verbose=False)

import os
gc_ = OUT + '/pugrid_tau%g_bins9.npz' % A.tau
Fp = Forward(cfg=cfg, tau_ns=0.0, verbose=False)
if os.path.exists(gc_):
    z = np.load(gc_)
    Fp.grid = dict(brain=z['brain'], bone=z['bone'], mean=z['mean'], cov=z['cov'])
    print('loaded cached pile-up grid', flush=True)
else:
    gb = np.concatenate([np.arange(0, 8, 1.), np.arange(8, 24, 2.)])
    Fp = Forward(cfg=cfg, tau_ns=A.tau, backend='mc', mc_frames=150,
                 grid_brain=gb, grid_bone=np.linspace(0, 8, 9), verbose=False)
    np.savez_compressed(gc_, **Fp.grid)

nch = g.nch
MI = np.zeros((9, nch, A.nview)); MD = np.zeros_like(MI); MP = np.zeros_like(MI)
chunk = max(1, A.maxrays // nch)
for s in range(0, A.nview, chunk):
    e = min(s + chunk, A.nview)
    vb, vbo = sb[:, s:e].ravel(), sbo[:, s:e].ravel()
    md, _, mi = F0.mean_cov(vb, vbo)
    mp, _, _ = Fp.mean_cov(vb, vbo)
    r = lambda m: m.T.reshape(9, nch, e - s)
    MI[:, :, s:e] = r(mi); MD[:, :, s:e] = r(md); MP[:, :, s:e] = r(mp)
print('counts %.0fs' % (time.time()-t0), flush=True)

def recon(sino):
    air = sino[5, 0]
    return g.fbp(np.ascontiguousarray(-np.log(np.maximum(sino, 1e-6) / max(air, 1e-9))))

fig, ax = plt.subplots(9, 4, figsize=(15, 32))
stats = []
for b in range(9):
    ri, rd, rp = recon(MI[b]), recon(MD[b]), recon(MP[b])
    v = np.percentile(ri, [2, 99])
    for j, (img, t) in enumerate([(ri, 'ideal'), (rd, '+sharing+escape'), (rp, '+pile-up')]):
        ax[b, j].imshow(img, cmap='gray', vmin=v[0], vmax=v[1]); ax[b, j].axis('off')
        ax[b, j].set_title('%s — bin %d (%s keV)' % (t, b+1, LBL[b]), fontsize=8)
    err = np.clip(100*(rp - ri)/np.maximum(np.abs(ri), 1e-4), -A.cap, A.cap)
    im = ax[b, 3].imshow(err, cmap='RdBu_r', norm=TwoSlopeNorm(0, -A.cap, A.cap))
    ax[b, 3].axis('off'); ax[b, 3].set_title('relative error (%)', fontsize=8)
    plt.colorbar(im, ax=ax[b, 3], fraction=.046)
    m = ri > np.percentile(ri, 60)
    stats.append((b+1, LBL[b], MI[b].sum(), MD[b].sum(), MP[b].sum(),
                  np.median(err[m])))
    print('  bin %d done %.0fs' % (b+1, time.time()-t0), flush=True)
fig.suptitle('Per-bin cascade: ideal / cross-talk / +pile-up / relative error', fontsize=14)
fig.tight_layout(); fig.savefig(FIGS + '/cascade_per_energy_bin.png', dpi=100, bbox_inches='tight')

print('\n%-4s %-10s %12s %12s %12s %12s' % ('bin','keV','ideal','+xtalk','+pileup','median err %'))
for b, l, a_, d_, p_, e_ in stats:
    print('%-4d %-10s %12.3e %12.3e %12.3e %+11.1f' % (b, l, a_, d_, p_, e_))
print('\nwrote figures/cascade_per_energy_bin.png (%.0fs)' % (time.time()-t0))

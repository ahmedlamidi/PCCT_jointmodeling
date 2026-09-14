"""Sinogram profile before/after charge sharing, for ALL 9 energy bins.

Same comparison as before_after/18_sinogram_profile_bin2.png, but at Morovati's
binning (20-109 keV in 10 keV steps) and every bin, not just one.

Only ONE view is needed for a profile, so this computes the rays directly from
the forward model instead of reading cascade.npz (which is 4-bin only). Seconds,
not hours.

    python3 fig_profile_9bin.py                 # view 360 of 720
    python3 fig_profile_9bin.py --view 0

Writes figures/before_after/19_sinogram_profile_9bin.png and, separately,
one PNG per bin under figures/before_after/bins9/.
"""
import argparse, os, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import FIGS
import pctk_compare as P
from pipeline import Forward
from projector import Geometry
from phantom import head

ap = argparse.ArgumentParser()
ap.add_argument('--nview', type=int, default=720)
ap.add_argument('--view', type=int, default=360)
ap.add_argument('--npix', type=int, default=512)
A = ap.parse_args()

ETH9 = [20., 30., 40., 50., 60., 70., 80., 90., 100.]
LBL = ['%g-%s' % (ETH9[i], ('%g' % ETH9[i+1]) if i < 8 else 'inf') for i in range(9)]
t0 = time.time()

brain, bone = head(A.npix)
g = Geometry(nview=A.nview, npix=A.npix)
sb, sbo = g.forward(brain), g.forward(bone)
vb, vbo = sb[:, A.view], sbo[:, A.view]
print('view %d: soft <= %.1f cm, bone <= %.1f cm  (%.0fs)'
      % (A.view, vb.max(), vbo.max(), time.time() - t0), flush=True)

cfg = dict(P.CFG); cfg['ETH'] = ETH9; cfg['FLUOR'] = 'off'   # charge sharing ONLY
F = Forward(cfg=cfg, tau_ns=0.0, verbose=False)
mu_d, _, mu_i = F.mean_cov(vb, vbo)          # (nch, 9) after / before
print('forward done (%.0fs)' % (time.time() - t0), flush=True)

ratio = mu_d / np.maximum(mu_i, 1e-9)
airm = np.zeros(len(vb), bool); airm[:2] = True              # pure air channels
inobj = mu_i[:, 0] < 0.5 * mu_i[airm, 0].mean()              # rays through the head

# Ratio statistics need a COUNT FLOOR. At 20-30 keV through 8 cm of bone the
# ideal count is ~1e-10, so the ratio there is 1e10 -- true but meaningless, and
# it is not a count anyone measures. Require at least MINC ideal counts, and
# quote 1st/99th percentiles rather than min/max.
MINC = 1e-2


def objmask(l):
    return inobj & (mu_i[:, l] > MINC)

SUB = os.path.join(FIGS, 'before_after')
os.makedirs(os.path.join(SUB, 'bins9'), exist_ok=True)


PFLOOR = 0.5           # half a count; below this nothing is measurable


def _m(v):
    return np.where(v >= PFLOOR, v, np.nan)


YMAX = float(max(np.nanmax(mu_i), np.nanmax(mu_d))) * 1.6


def draw(ax, l):
    a, b = mu_i[:, l], mu_d[:, l]
    ax.semilogy(_m(a), lw=1.5, color='C0', label='before (ideal)')
    ax.semilogy(_m(b), lw=1.3, color='C3', label='after (sharing)')
    ax.set_ylim(PFLOOR, YMAX)
    ax.set_title('bin %d  (%s keV)' % (l + 1, LBL[l]), fontsize=10)
    ax.set_xlabel('channel'); ax.set_ylabel('counts')
    ax.grid(alpha=.3); ax.legend(fontsize=7)


fig, axs = plt.subplots(3, 3, figsize=(16.5, 13), sharey=True)
for l in range(9):
    draw(axs[l // 3][l % 3], l)
fig.suptitle('Charge sharing, before vs after — all 9 energy bins, view %d' % A.view,
             fontsize=14)
fig.tight_layout()
p = os.path.join(SUB, '19_sinogram_profile_9bin.png')
fig.savefig(p, dpi=115, bbox_inches='tight'); plt.close(fig)
print('wrote', p)

for l in range(9):
    f1 = plt.figure(figsize=(6.2, 5.0))
    draw(f1.add_subplot(111), l)
    f1.tight_layout()
    fp = os.path.join(SUB, 'bins9', 'profile_bin%d.png' % (l + 1))
    f1.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(f1)
print('wrote', os.path.join(SUB, 'bins9'), '(9 files)')

# ---- one panel with every ratio ---------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 5.6))
cm = plt.cm.viridis(np.linspace(0, .92, 9))
for l in range(9):
    ax.semilogy(ratio[:, l], lw=1.5, color=cm[l], label='bin %d (%s)' % (l + 1, LBL[l]))
ax.axhline(1.0, color='k', lw=1.3)
ax.set_xlabel('channel'); ax.set_ylabel('counts after / counts before')
ax.set_title('Ratio per bin, all 9 (view %d)' % A.view, fontsize=11)
ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=.3)
fig.tight_layout()
p = os.path.join(SUB, '20_sinogram_ratio_9bin.png')
fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
print('wrote', p)

print('\nRatio statistics over rays through the head with >= %g ideal counts,' % MINC)
print('quoted as 1st-99th percentile (min/max explode where the ideal count ~ 0).\n')
print('%-5s %-9s %8s %10s %10s %8s %8s'
      % ('bin', 'keV', 'air', 'obj p1', 'obj p99', 'cross', 'n rays'))
for l in range(9):
    m = objmask(l); ro = ratio[m, l]
    r_air = mu_d[airm, l].mean() / mu_i[airm, l].mean()
    if ro.size == 0:
        print('%-5d %-9s %7.0f%% %10s %10s %8s %8d' % (l+1, LBL[l], 100*(r_air-1), '-', '-', '', 0))
        continue
    lo, hi = np.percentile(ro, [1, 99])
    cr = 'YES' if (min(lo, r_air) < 1 < max(hi, r_air)) else ''
    print('%-5d %-9s %7.0f%% %9.0f%% %9.0f%% %8s %8d'
          % (l + 1, LBL[l], 100*(r_air-1), 100*(lo-1), 100*(hi-1), cr, int(m.sum())))

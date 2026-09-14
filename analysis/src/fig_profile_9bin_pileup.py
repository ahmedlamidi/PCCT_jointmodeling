"""Profile per energy bin: ideal / +pile-up / +charge sharing / +both.

Same layout as fig_profile_9bin.py, but four curves per panel so each effect can
be seen alone and in combination:

    1 ideal                    perfect spectrometer, same QE
    2 + pile-up only           ideal response, then the readout dead time
    3 + charge sharing only    PcTK FLUOR='off', no dead time
    4 + charge sharing + pile-up

Stage 2 is an ablation: pile-up applied to an IDEAL deposited spectrum. It is
not a detector anyone builds, but it isolates what the readout does from what
the sensor does.

Pile-up is gridded on (soft, bone) and interpolated, so the grid build dominates
the runtime (~2 s per point, ~180 points, per model).

    python3 fig_profile_9bin_pileup.py --tau 30

Writes figures/before_after/21_profile_9bin_pileup.png plus one PNG per bin
under figures/before_after/bins9_pileup/.
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
ap.add_argument('--tau', type=float, default=30.0, help='dead time, ns')
ap.add_argument('--t_p', type=float, default=10.0, help='peaking time, ns')
ap.add_argument('--T', type=float, default=25.0, help='pulse length, ns (must be <= tau)')
ap.add_argument('--mcframes', type=int, default=100)
A = ap.parse_args()
if A.T > A.tau:
    raise SystemExit('pulse length T=%g ns exceeds dead time tau=%g ns: every pulse '
                     'would retrigger on its own tail.' % (A.T, A.tau))

ETH9 = [20., 30., 40., 50., 60., 70., 80., 90., 100.]
LBL = ['%g-%s' % (ETH9[i], ('%g' % ETH9[i+1]) if i < 8 else 'inf') for i in range(9)]
PFLOOR = 0.5
t0 = time.time()

brain, bone = head(A.npix)
g = Geometry(nview=A.nview, npix=A.npix)
vb = g.forward(brain)[:, A.view]
vbo = g.forward(bone)[:, A.view]
print('view %d: soft <= %.1f cm, bone <= %.1f cm  (%.0fs)'
      % (A.view, vb.max(), vbo.max(), time.time() - t0), flush=True)

# grid must span this view's paths
GB = np.concatenate([np.arange(0, 10, 1.), np.arange(10, float(vb.max()) + 4, 2.)])
GC = np.linspace(0, max(9.0, float(vbo.max()) * 1.1), 10)
print('pile-up grid %d x %d = %d points per model' % (len(GB), len(GC), len(GB) * len(GC)), flush=True)

cfg = dict(P.CFG); cfg['ETH'] = ETH9; cfg['FLUOR'] = 'off'   # charge sharing only


def with_pileup(ideal_response):
    """Forward with the pile-up grid built. ideal_response swaps PcTK's response
    for the perfect one, which isolates pile-up from charge sharing."""
    F = Forward(cfg=cfg, tau_ns=0.0, verbose=False)          # no grid yet
    if ideal_response:
        F.R = F.Rideal
        F.Wf = F.Wi
    F.tau = A.tau * 1e-9; F.t_p = A.t_p * 1e-9; F.T = A.T * 1e-9
    F.backend, F.mc_frames = 'mc', A.mcframes
    F._build_grid(GB, GC, verbose=True)
    return F


F0 = Forward(cfg=cfg, tau_ns=0.0, verbose=False)             # no pile-up
mu_d, _, mu_i = F0.mean_cov(vb, vbo)                         # sharing / ideal
print('no-pile-up stages done (%.0fs)' % (time.time() - t0), flush=True)

print('building grid: ideal response + pile-up ...', flush=True)
mu_pu_ideal, _, _ = with_pileup(True).mean_cov(vb, vbo)
print('building grid: charge sharing + pile-up ...', flush=True)
mu_pu_cs, _, _ = with_pileup(False).mean_cov(vb, vbo)
print('all stages done (%.0fs)' % (time.time() - t0), flush=True)

STAGES = [('ideal', mu_i, 'C0', 1.6),
          ('+ pile-up', mu_pu_ideal, 'C2', 1.2),
          ('+ charge sharing', mu_d, 'C3', 1.2),
          ('+ sharing + pile-up', mu_pu_cs, '0.15', 1.5)]

SUB = os.path.join(FIGS, 'before_after')
os.makedirs(os.path.join(SUB, 'bins9_pileup'), exist_ok=True)
YMAX = max(float(np.nanmax(m)) for _, m, _, _ in STAGES) * 1.6


def _m(v):
    return np.where(v >= PFLOOR, v, np.nan)


def draw(ax, l):
    for nm, M, col, lw in STAGES:
        ax.semilogy(_m(M[:, l]), lw=lw, color=col, label=nm)
    ax.set_ylim(PFLOOR, YMAX)
    ax.set_title('bin %d  (%s keV)' % (l + 1, LBL[l]), fontsize=10)
    ax.set_xlabel('channel'); ax.set_ylabel('counts')
    ax.grid(alpha=.3); ax.legend(fontsize=7)


fig, axs = plt.subplots(3, 3, figsize=(16.5, 13), sharey=True)
for l in range(9):
    draw(axs[l // 3][l % 3], l)
fig.suptitle('Charge sharing and pile-up — all 9 energy bins, view %d  '
             '($\\tau$=%g ns, $t_p$=%g ns, T=%g ns)' % (A.view, A.tau, A.t_p, A.T),
             fontsize=14)
fig.tight_layout()
p = os.path.join(SUB, '21_profile_9bin_pileup.png')
fig.savefig(p, dpi=115, bbox_inches='tight'); plt.close(fig)
print('wrote', p)

for l in range(9):
    f1 = plt.figure(figsize=(6.2, 5.0))
    draw(f1.add_subplot(111), l)
    f1.tight_layout()
    f1.savefig(os.path.join(SUB, 'bins9_pileup', 'profile_bin%d.png' % (l + 1)),
               dpi=150, bbox_inches='tight'); plt.close(f1)
print('wrote', os.path.join(SUB, 'bins9_pileup'), '(9 files)')

np.savez_compressed(os.path.join(FIGS, '..', 'outputs', 'profile9_pileup.npz'),
                    vb=vb, vbo=vbo, ideal=mu_i, pileup=mu_pu_ideal,
                    sharing=mu_d, both=mu_pu_cs, eth=ETH9)

airm = np.zeros(len(vb), bool); airm[:2] = True
inobj = mu_i[:, 0] < 0.5 * mu_i[airm, 0].mean()
print('\nchange vs ideal, in AIR (%):')
print('%-5s %-9s %10s %10s %10s' % ('bin', 'keV', 'pile-up', 'sharing', 'both'))
for l in range(9):
    r = [100 * (M[airm, l].mean() / mu_i[airm, l].mean() - 1) for _, M, _, _ in STAGES[1:]]
    print('%-5d %-9s %9.0f%% %9.0f%% %9.0f%%' % (l + 1, LBL[l], *r))
print('\nchange vs ideal, THROUGH THE HEAD (median over rays with >=0.01 ideal counts, %):')
print('%-5s %-9s %10s %10s %10s' % ('bin', 'keV', 'pile-up', 'sharing', 'both'))
for l in range(9):
    m = inobj & (mu_i[:, l] > 1e-2)
    if m.sum() == 0:
        print('%-5d %-9s %10s %10s %10s' % (l + 1, LBL[l], '-', '-', '-')); continue
    r = [100 * (np.median(M[m, l] / mu_i[m, l]) - 1) for _, M, _, _ in STAGES[1:]]
    print('%-5d %-9s %9.0f%% %9.0f%% %9.0f%%' % (l + 1, LBL[l], *r))

"""What pile-up adds ON TOP of charge sharing: sharing alone vs sharing + pile-up.

Two curves per panel, so the marginal effect of the readout is not hidden behind
the much larger charge-sharing distortion. Reads the arrays cached by
fig_profile_9bin_pileup.py, so it runs in seconds.

    python3 fig_profile_9bin_pileup.py --tau 30     # once, ~12 min, writes the npz
    python3 fig_profile_9bin_marginal.py            # this, instant

Writes figures/before_after/22_profile_9bin_marginal.png,
       figures/before_after/23_ratio_9bin_marginal.png,
   and figures/before_after/bins9_marginal/profile_bin{1..9}.png
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import FIGS, OUT

D = np.load(os.path.join(OUT, 'profile9_pileup.npz'))
sharing, both, ideal = D['sharing'], D['both'], D['ideal']
ETH9 = list(D['eth'])
LBL = ['%g-%s' % (ETH9[i], ('%g' % ETH9[i+1]) if i < 8 else 'inf') for i in range(9)]
PFLOOR = 0.5
YMAX = max(float(np.nanmax(sharing)), float(np.nanmax(both))) * 1.6

SUB = os.path.join(FIGS, 'before_after')
os.makedirs(os.path.join(SUB, 'bins9_marginal'), exist_ok=True)


def _m(v):
    return np.where(v >= PFLOOR, v, np.nan)


def draw(ax, l):
    ax.semilogy(_m(sharing[:, l]), lw=1.6, color='C3', label='charge sharing')
    ax.semilogy(_m(both[:, l]), lw=1.4, color='0.15', label='charge sharing + pile-up')
    ax.set_ylim(PFLOOR, YMAX)
    ax.set_title('bin %d  (%s keV)' % (l + 1, LBL[l]), fontsize=10)
    ax.set_xlabel('channel'); ax.set_ylabel('counts')
    ax.grid(alpha=.3); ax.legend(fontsize=7)


fig, axs = plt.subplots(3, 3, figsize=(16.5, 13), sharey=True)
for l in range(9):
    draw(axs[l // 3][l % 3], l)
fig.suptitle('What pile-up adds on top of charge sharing — view 360, '
             '$\\tau$=30 ns', fontsize=14)
fig.tight_layout()
p = os.path.join(SUB, '22_profile_9bin_marginal.png')
fig.savefig(p, dpi=115, bbox_inches='tight'); plt.close(fig)
print('wrote', p)

for l in range(9):
    f1 = plt.figure(figsize=(6.2, 5.0))
    draw(f1.add_subplot(111), l)
    f1.tight_layout()
    f1.savefig(os.path.join(SUB, 'bins9_marginal', 'profile_bin%d.png' % (l + 1)),
               dpi=150, bbox_inches='tight'); plt.close(f1)
print('wrote', os.path.join(SUB, 'bins9_marginal'), '(9 files)')

# ---- ratio: pile-up's marginal effect, all bins on one axis ----------
r = both / np.maximum(sharing, 1e-9)
fig, ax = plt.subplots(figsize=(8.2, 5.6))
cm = plt.cm.viridis(np.linspace(0, .92, 9))
for l in range(9):
    ax.semilogy(r[:, l], lw=1.5, color=cm[l], label='bin %d (%s)' % (l + 1, LBL[l]))
ax.axhline(1.0, color='k', lw=1.3)
ax.set_xlabel('channel'); ax.set_ylabel('(sharing + pile-up) / sharing')
ax.set_title('Marginal effect of pile-up, view 360', fontsize=11)
ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=.3)
fig.tight_layout()
p = os.path.join(SUB, '23_ratio_9bin_marginal.png')
fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
print('wrote', p)

airm = np.zeros(sharing.shape[0], bool); airm[:2] = True
inobj = ideal[:, 0] < 0.5 * ideal[airm, 0].mean()
print('\npile-up ON TOP OF charge sharing (%% change vs sharing alone)')
print('%-5s %-9s %10s %12s' % ('bin', 'keV', 'air', 'through head'))
for l in range(9):
    a = 100 * (both[airm, l].mean() / sharing[airm, l].mean() - 1)
    m = inobj & (sharing[:, l] > 1e-2)
    h = 100 * (np.median(both[m, l] / sharing[m, l]) - 1) if m.sum() else np.nan
    print('%-5d %-9s %9.0f%% %11.0f%%' % (l + 1, LBL[l], a, h))

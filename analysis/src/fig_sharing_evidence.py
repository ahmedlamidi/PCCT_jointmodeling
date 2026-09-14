"""One figure: is charge sharing present? Three panels, no explanation needed.

A  where a single monoenergetic photon's energy lands across the 3x3 neighbourhood
B  the energy that photon is RECORDED at, vs where it should be
C  how much energy leaves the struck pixel, across the spectrum
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '.')
from paths import FIGS
import pctk_compare as P

ap = argparse.ArgumentParser()
ap.add_argument('--E', type=int, default=100, help='incident photon energy, keV')
ap.add_argument('--floor', type=float, default=6.0, help='ignore bins below this (noise floor)')
A = ap.parse_args()

d = P.load(); vEo, vE1 = d['vEo'], d['vE1']
diag = d['diagE'].reshape(170, 9, 191)
q0 = d['q'][0].reshape(170, 9, 191)
iE = int(np.where(vE1 == A.E)[0][0])
m = vEo >= A.floor

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))

# ---- A: 3x3 energy map -------------------------------------------------
e_pix = (diag[iE][:, m] * vEo[m]).sum(1)             # keV deposited per pixel
G = e_pix.reshape(3, 3)
im = ax[0].imshow(G, cmap='inferno')
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, '%.1f\nkeV' % G[i, j], ha='center', va='center', fontsize=13,
                   color='w' if G[i, j] < G.max() * .55 else 'k', fontweight='bold')
ax[0].set_xticks([]); ax[0].set_yticks([])
out = 100 * (1 - G[1, 1] / G.sum())
ax[0].set_title('A.  Where one %d keV photon\'s energy lands\n'
                'photon strikes the CENTRE pixel — %.0f%% escapes to neighbours'
                % (A.E, out), fontsize=12)
plt.colorbar(im, ax=ax[0], fraction=.046, label='keV deposited')

# ---- B: recorded spectrum ---------------------------------------------
s_all = diag[iE][:, :].sum(0)
s_ctr = diag[iE][4]
ax[1].semilogy(vEo, np.maximum(s_ctr, 1e-6), lw=2.0, color='C0',
               label='struck pixel')
ax[1].semilogy(vEo, np.maximum(s_all, 1e-6), lw=1.6, color='C1', ls='--',
               label='all 9 pixels summed')
ax[1].axvline(A.E, color='k', lw=2.2, label='ideal detector: a single line at %d keV' % A.E)
ax[1].axvspan(A.E - 27.47, A.E - 23.17, color='C2', alpha=.25,
              label='K-escape ($E-K_\\alpha$)')
ax[1].set_xlim(0, A.E + 25); ax[1].set_ylim(1e-4, 1)
ax[1].set_xlabel('recorded energy (keV)'); ax[1].set_ylabel('counts / incident photon / keV')
ax[1].set_title('B.  A perfect detector records one line.\nThis one records a smear.',
                fontsize=12)
ax[1].legend(fontsize=9, loc='upper left'); ax[1].grid(alpha=.3)

# ---- C: energy leaving the struck pixel, vs energy ---------------------
Es = np.arange(25, 141)
ff, f0 = [], []
for E in Es:
    i = int(np.where(vE1 == E)[0][0])
    e = (diag[i][:, m] * vEo[m]).sum(1); ff.append(100 * (1 - e[4] / e.sum()))
    e = (q0[i][:, m] * vEo[m]).sum(1); f0.append(100 * (1 - e[4] / e.sum()))
ax[2].plot(Es, ff, lw=2.2, color='C3', label='total (sharing + fluorescence)')
ax[2].plot(Es, f0, lw=2.0, color='C0', label='charge sharing alone')
ax[2].fill_between(Es, f0, ff, color='C2', alpha=.25, label='fluorescence transport')
ax[2].axhline(8.85, color='k', ls=':', lw=1.8,
              label='hard-sphere geometry ($r_0$=24 µm in 225 µm)')
ax[2].axvline(26.71, color='.5', lw=1.2, ls='--')
ax[2].text(27.5, 4, 'Cd K-edge', rotation=90, fontsize=9, color='.4')
ax[2].set_xlabel('incident photon energy (keV)')
ax[2].set_ylabel('% of energy leaving the struck pixel')
ax[2].set_title('C.  It happens at every energy,\nand more than geometry alone predicts',
                fontsize=12)
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3); ax[2].set_ylim(0, None)

fig.tight_layout()
fig.savefig(FIGS + '/physics_charge_sharing_evidence.png', dpi=140, bbox_inches='tight')
print('%d keV photon strikes the centre pixel:' % A.E)
print('  centre    %.1f keV' % G[1, 1])
print('  4 edges   %.1f keV' % G[[0, 1, 1, 2], [1, 0, 2, 1]].sum())
print('  4 corners %.1f keV' % G[[0, 0, 2, 2], [0, 2, 0, 2]].sum())
print('  -> %.0f%% of the deposited energy is recorded in a pixel the photon never hit' % out)
print('\nwrote figures/physics_charge_sharing_evidence.png')

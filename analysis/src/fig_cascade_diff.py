"""Cascade shown as DIFFERENCE maps: what each effect adds, and where.

Side-by-side absolute panels hide pile-up almost entirely through the patient,
because it varies by two orders of magnitude across one projection. A signed
difference against the previous stage shows both magnitude and location.

Row 1  sinogram, absolute (log)          -- reference
Row 2  sinogram, delta vs previous stage -- signed, what this effect ADDED
Row 3  reconstruction, absolute
Row 4  reconstruction, delta vs previous stage

Reads outputs/cascade.npz (written by fig_cascade.py). No recomputation.
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from paths import FIGS, OUT

ap = argparse.ArgumentParser()
ap.add_argument('--bin', type=int, default=1, choices=[1, 4])
ap.add_argument('--pct', type=float, default=99.0, help='percentile for diff colour limits')
A = ap.parse_args()

z = np.load(OUT + '/cascade.npz')
S = ['1 ideal', '2 + charge sharing', '3 + K-escape', '4 + pile-up', '5 + noise']
LBL = {1: '20-50 keV', 4: '80+ keV'}[A.bin]
bi = A.bin - 1

sino = [z['sino_' + s][bi] for s in S]                 # (nch, nview)
rec = [z['%s_bin%d' % (s, A.bin)] for s in S]

# one display window per row, taken from the ideal panel
sv = (max(sino[0].min(), 1e-2), sino[0].max())
rv = np.percentile(rec[0], [1, 99.5])

fig, ax = plt.subplots(4, 5, figsize=(21, 15.5))
for c, s in enumerate(S):
    # --- row 1: sinogram, absolute
    ax[0, c].imshow(sino[c].T, aspect='auto', cmap='gray', norm=LogNorm(*sv),
                    extent=[0, sino[c].shape[0], sino[c].shape[1], 0])
    ax[0, c].set_title(s, fontsize=11)
    if c == 0: ax[0, c].set_ylabel('sinogram\nview', fontsize=10)

    # --- row 2: sinogram, delta vs previous
    if c == 0:
        ax[1, c].axis('off')
        ax[1, c].text(.5, .5, '(nothing before ideal)', ha='center', va='center',
                      fontsize=10, color='.5', transform=ax[1, c].transAxes)
    else:
        d = sino[c] - sino[c - 1]
        lim = np.percentile(np.abs(d), A.pct) or 1.0
        im = ax[1, c].imshow(d.T, aspect='auto', cmap='RdBu_r',
                             norm=TwoSlopeNorm(0, -lim, lim),
                             extent=[0, d.shape[0], d.shape[1], 0])
        ax[1, c].set_title('delta counts: %+.0f to %+.0f' % (d.min(), d.max()), fontsize=9)
        plt.colorbar(im, ax=ax[1, c], fraction=.046)
    if c == 0: ax[1, c].set_ylabel('sinogram delta', fontsize=10)

    # --- row 3: reconstruction, absolute
    ax[2, c].imshow(rec[c], cmap='gray', vmin=rv[0], vmax=rv[1]); ax[2, c].axis('off')
    ax[2, c].set_title('reconstruction', fontsize=9)

    # --- row 4: reconstruction, delta vs previous
    if c == 0:
        ax[3, c].axis('off')
    else:
        d = rec[c] - rec[c - 1]
        lim = np.percentile(np.abs(d), A.pct) or 1.0
        im = ax[3, c].imshow(d, cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim))
        ax[3, c].set_title('recon delta', fontsize=9)
        plt.colorbar(im, ax=ax[3, c], fraction=.046)
    ax[3, c].axis('off')

for r in (0,):
    for c in range(5):
        ax[r, c].set_xlabel('channel', fontsize=8)
fig.suptitle('Cascade as difference maps, bin %d (%s).  Red = this effect ADDED counts, '
             'blue = removed them.' % (A.bin, LBL), fontsize=13)
fig.tight_layout()
out = FIGS + '/cascade_delta_counts_bin%d.png' % A.bin
fig.savefig(out, dpi=115, bbox_inches='tight')

print('bin %d (%s)   net counts per stage and delta:' % (A.bin, LBL))
prev = None
for c, s in enumerate(S):
    tot = sino[c].sum()
    d = '' if prev is None else '   delta %+12.0f (%+6.1f%%)' % (tot - prev, 100*(tot-prev)/prev)
    print('  %-20s %14.0f%s' % (s, tot, d))
    prev = tot
print('\nwrote %s' % out.split('/figures/')[-1])

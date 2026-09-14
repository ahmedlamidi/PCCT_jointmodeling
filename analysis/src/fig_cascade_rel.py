"""Cascade as RELATIVE difference maps.

fig_cascade_diff.py plotted absolute count deltas. Counts span five decades
(12,000 photons in air, ~0.4% transmission through a head), so absolute deltas
are dominated by the air gap at the detector edges and the effect looks confined
there. It is not. In relative terms every effect is STRONGER through the object.

Row 1  relative delta vs previous stage, %      -- what each effect adds
Row 2  cumulative relative error vs ideal, %    -- how far from truth we are
Row 3  reconstruction, relative delta vs ideal

Colour limits are symmetric percentiles so the object interior is readable.
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from paths import FIGS, OUT

ap = argparse.ArgumentParser()
ap.add_argument('--bin', type=int, default=1, choices=[1, 4])
ap.add_argument('--cap', type=float, default=300.0, help='colour limit, %')
A = ap.parse_args()

z = np.load(OUT + '/cascade.npz')
S = ['1 ideal', '2 + charge sharing', '3 + K-escape', '4 + pile-up', '5 + noise']
LBL = {1: '20-50 keV', 4: '80+ keV'}[A.bin]
bi = A.bin - 1
sino = [z['sino_' + s][bi] for s in S]
rec = [z['%s_bin%d' % (s, A.bin)] for s in S]

def rel(a, b):
    return 100.0 * (a - b) / np.maximum(np.abs(b), 1e-3)

fig, ax = plt.subplots(3, 4, figsize=(19, 12))
for c in range(1, 5):
    j = c - 1
    ext = [0, sino[0].shape[0], sino[0].shape[1], 0]

    d = np.clip(rel(sino[c], sino[c - 1]), -A.cap, A.cap)
    im = ax[0, j].imshow(d.T, aspect='auto', cmap='RdBu_r',
                         norm=TwoSlopeNorm(0, -A.cap, A.cap), extent=ext)
    ax[0, j].set_title('%s\nvs previous stage' % S[c], fontsize=10)
    plt.colorbar(im, ax=ax[0, j], fraction=.046, label='%')

    d = np.clip(rel(sino[c], sino[0]), -A.cap, A.cap)
    im = ax[1, j].imshow(d.T, aspect='auto', cmap='RdBu_r',
                         norm=TwoSlopeNorm(0, -A.cap, A.cap), extent=ext)
    ax[1, j].set_title('cumulative vs ideal', fontsize=10)
    plt.colorbar(im, ax=ax[1, j], fraction=.046, label='%')

    d = np.clip(rel(rec[c], rec[0]), -A.cap, A.cap)
    im = ax[2, j].imshow(d, cmap='RdBu_r', norm=TwoSlopeNorm(0, -A.cap, A.cap))
    ax[2, j].set_title('reconstruction vs ideal', fontsize=10); ax[2, j].axis('off')
    plt.colorbar(im, ax=ax[2, j], fraction=.046, label='%')

ax[0, 0].set_ylabel('view'); ax[1, 0].set_ylabel('view')
for j in range(4):
    ax[1, j].set_xlabel('channel')
fig.suptitle('Cascade as RELATIVE change, bin %d (%s). Colour capped at +/-%.0f%%.  '
             'Red = counts added, blue = removed.' % (A.bin, LBL, A.cap), fontsize=13)
fig.tight_layout()
out = FIGS + '/cascade_delta_percent_bin%d.png' % A.bin
fig.savefig(out, dpi=115, bbox_inches='tight')

# quantify: object interior vs air edges
nch = sino[0].shape[0]
edge = np.r_[0:180, nch-180:nch]
obj = np.r_[600:1250]
print('bin %d (%s)   median |relative change| vs ideal' % (A.bin, LBL))
print('  %-20s %14s %14s' % ('stage', 'air edges', 'object centre'))
for c in range(1, 5):
    r = np.abs(rel(sino[c], sino[0]))
    print('  %-20s %13.1f%% %13.1f%%' % (S[c], np.median(r[edge]), np.median(r[obj])))
print('\nwrote %s' % out.split('/figures/')[-1])

"""Phantom -> sinogram -> signal -> reconstruction, at Morovati's 9-bin binning.

Produces the figure set that mirrors what their paper shows:

    data_phantom.png            the two material maps and a composite
    data_sinogram_9bin.png      line integrals, then counts per energy bin
    data_signal_at_rays.png     what the detector reports: spectra at chosen
                         rays (ideal vs distorted), a channel profile, and noise
    recon_9bin_ideal_vs_pctk.png  per-energy-bin FBP, ideal vs distorted

Default binning is 20-109 keV in 10 keV steps (9 bins), matching
Morovati et al., Phys Med Biol 2025, doi 10.1088/1361-6560/adaf71.

    python3 make_paper_figs.py --nview 720 --bins 9
    python3 make_paper_figs.py --nview 2000 --bins 4     # PcTK's own binning

FIRST RUN at 9 bins builds the rebinned covariance from the 4 GB nCovE
(~7-10 min, cached afterwards). Run  python3 rebin_cov.py --validate  once first.
"""
import argparse, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from paths import FIGS, OUT
import pctk_compare as P
from pipeline import Forward
from projector import Geometry
from phantom import head

ap = argparse.ArgumentParser()
ap.add_argument('--nview', type=int, default=720)
ap.add_argument('--npix', type=int, default=512)
ap.add_argument('--bins', type=int, choices=[4, 9], default=9)
ap.add_argument('--chunk', type=int, default=100)
ap.add_argument('--tau', type=float, default=0.0)
A = ap.parse_args()

ETH = ([20., 50., 65., 80.] if A.bins == 4
       else [20., 30., 40., 50., 60., 70., 80., 90., 100.])
LBL = ['%g-%s' % (ETH[i], ('%g' % ETH[i + 1]) if i < len(ETH) - 1 else 'inf')
       for i in range(len(ETH))]
t0 = time.time()

# ---------------------------------------------------------------- phantom
brain, bone = head(A.npix)
print('phantom  brain max %.2f  bone max %.2f' % (brain.max(), bone.max()), flush=True)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
ax[0].imshow(brain, cmap='gray'); ax[0].set_title('soft tissue fraction')
ax[1].imshow(bone, cmap='gray');  ax[1].set_title('bone fraction')
rgb = np.stack([np.clip(bone, 0, 1), np.clip(brain, 0, 1), np.zeros_like(bone)], -1)
ax[2].imshow(rgb); ax[2].set_title('composite (red=bone, green=soft)')
for a_ in ax: a_.axis('off')
fig.suptitle('Two-material Shepp-Logan head phantom, %d x %d' % (A.npix, A.npix))
fig.tight_layout(); fig.savefig(FIGS + '/data_phantom.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- sinogram
g = Geometry(nview=A.nview, npix=A.npix)
sb = g.forward(brain); sbo = g.forward(bone)
print('line integrals  brain max %.1f cm  bone max %.1f cm   %.0fs'
      % (sb.max(), sbo.max(), time.time() - t0), flush=True)

cfg = dict(P.CFG); cfg['ETH'] = ETH
F = Forward(cfg=cfg, tau_ns=A.tau)
Nl = F.Nl
rng = np.random.default_rng(0)

X = np.zeros((Nl, g.nch, g.nview))          # distorted, noisy
MD = np.zeros_like(X)                       # distorted mean
MI = np.zeros_like(X)                       # ideal mean
for s in range(0, g.nview, A.chunk):
    e = min(s + A.chunk, g.nview)
    vb = sb[:, s:e].ravel(); vbo = sbo[:, s:e].ravel()
    yd, yi, md, mi = F.sample(vb, vbo, rng)
    X[:, :, s:e] = yd.T.reshape(Nl, g.nch, e - s)
    MD[:, :, s:e] = md.T.reshape(Nl, g.nch, e - s)
    MI[:, :, s:e] = mi.T.reshape(Nl, g.nch, e - s)
    print('  counts views %4d/%d  %.0fs' % (e, g.nview, time.time() - t0), flush=True)

show = [0, Nl // 2, Nl - 1]
fig, ax = plt.subplots(1, 2 + len(show), figsize=(4 * (2 + len(show)), 4.2))
for k, (S, t) in enumerate([(sb, 'soft tissue'), (sbo, 'bone')]):
    im = ax[k].imshow(S.T, aspect='auto', cmap='magma', extent=[0, g.nch, g.nview, 0])
    ax[k].set_title('line integral, %s (cm)' % t, fontsize=9); plt.colorbar(im, ax=ax[k])
for k, l in enumerate(show):
    im = ax[2 + k].imshow(X[l].T, aspect='auto', cmap='gray',
                          norm=LogNorm(max(X[l].min(), 1), X[l].max()),
                          extent=[0, g.nch, g.nview, 0])
    ax[2 + k].set_title('counts, bin %d (%s keV)' % (l + 1, LBL[l]), fontsize=9)
    plt.colorbar(im, ax=ax[2 + k])
for a_ in ax: a_.set_xlabel('channel')
ax[0].set_ylabel('view')
fig.suptitle('Sinograms: material line integrals, then recorded counts per energy bin')
fig.tight_layout(); fig.savefig(FIGS + '/data_sinogram_9bin.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- the signal
cc = g.nch // 2
rays = [('air', 20), ('skull edge', cc - 430), ('centre', cc)]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
w = np.arange(Nl)
for nm, ch in rays:
    ax[0].plot(w, MI[:, ch, 0], 's--', alpha=.6, label='%s ideal' % nm)
    ax[0].plot(w, MD[:, ch, 0], 'o-', label='%s PcTK' % nm)
ax[0].set_yscale('log'); ax[0].set_xticks(w); ax[0].set_xticklabels(LBL, rotation=45, fontsize=7)
ax[0].set_xlabel('energy bin (keV)'); ax[0].set_ylabel('counts / pixel / view')
ax[0].set_title('recorded spectrum at three rays'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

for l in show:
    ax[1].semilogy(MD[l, :, 0], lw=1.1, label='PcTK bin %d' % (l + 1))
    ax[1].semilogy(MI[l, :, 0], lw=.9, ls='--', alpha=.6)
ax[1].set_xlabel('channel'); ax[1].set_ylabel('counts'); ax[1].legend(fontsize=7)
ax[1].set_title('view 1 profile (solid PcTK, dashed ideal)'); ax[1].grid(alpha=.3)

sl = slice(cc - 40, cc + 40)
ax[2].plot(np.arange(sl.start, sl.stop), MD[0, sl, 0], 'k-', lw=2, label='mean')
ax[2].plot(np.arange(sl.start, sl.stop), X[0, sl, 0], 'r-', lw=1, label='one realisation')
ax[2].set_xlabel('channel'); ax[2].set_ylabel('counts, bin 1')
ax[2].set_title('noise, bin 1'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
fig.suptitle('What the detector actually reports')
fig.tight_layout(); fig.savefig(FIGS + '/data_signal_at_rays.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- recon
air_d = MD[:, 5, 0]; air_i = MI[:, 5, 0]
fig, ax = plt.subplots(2, len(show), figsize=(4.3 * len(show), 8.4))
recs = {}
for k, l in enumerate(show):
    for r, (D, air, t) in enumerate([(MI, air_i, 'ideal'), (MD, air_d, 'PcTK')]):
        gi = -np.log(np.maximum(D[l], 1e-3) / air[l])
        im = g.fbp(np.ascontiguousarray(gi))
        recs['%s_bin%d' % (t, l + 1)] = im
        v = np.percentile(im, [1, 99.5])
        ax[r, k].imshow(im, cmap='gray', vmin=v[0], vmax=v[1]); ax[r, k].axis('off')
        ax[r, k].set_title('%s — bin %d (%s keV)' % (t, l + 1, LBL[l]), fontsize=9)
    print('  recon bin %d  %.0fs' % (l + 1, time.time() - t0), flush=True)
np.savez_compressed(OUT + '/paper_recons.npz', **recs)
fig.suptitle('Per-energy-bin FBP: ideal detector (top) vs PcTK detector (bottom)')
fig.tight_layout(); fig.savefig(FIGS + '/recon_9bin_ideal_vs_pctk.png', dpi=130, bbox_inches='tight')
plt.close(fig)

print('\nwrote data_phantom, data_sinogram_9bin, data_signal_at_rays, recon_9bin_ideal_vs_pctk  (%.0fs)'
      % (time.time() - t0))

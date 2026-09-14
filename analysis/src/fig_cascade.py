"""Cumulative physics cascade: sinogram and reconstruction, one effect at a time.

    1 ideal            perfect spectrometer, same QE, noise free
    2 + charge sharing PcTK with fluorescence OFF          (FLUOR='off')
    3 + K-escape       PcTK full                            (FLUOR='full')
    4 + pile-up        rate-dependent stage, MC backend
    5 + noise          correlated realisation of stage 4

Stages 1-4 are noise free, so any difference between them is bias, not luck.
Stage 5 adds the realisation on top of stage 4.

Line integrals are air-normalised per stage, as a real scan would be, so the
figure shows what survives after the air calibration a scanner already does.

    python3 fig_cascade.py --nview 720 --tau 30

NOTE the FBP constant is off by ~1.85 (see PIPELINE.md). It is the same constant
in every panel, so the comparison is unaffected; absolute values are not.
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
ap.add_argument('--tau', type=float, default=30.0, help='dead time, ns')
ap.add_argument('--chunk', type=int, default=60)
ap.add_argument('--bins', type=int, default=4, choices=[4, 9])
ap.add_argument('--mcframes', type=int, default=120)
A = ap.parse_args()

ETH = ([20., 50., 65., 80.] if A.bins == 4
       else [20., 30., 40., 50., 60., 70., 80., 90., 100.])
LBL = ['%g-%s' % (ETH[i], ('%g' % ETH[i+1]) if i < len(ETH)-1 else 'inf')
       for i in range(len(ETH))]
t0 = time.time()

brain, bone = head(A.npix)
g = Geometry(nview=A.nview, npix=A.npix)
sb, sbo = g.forward(brain), g.forward(bone)
print('projected  soft<=%.1f cm  bone<=%.1f cm  (%.0fs)' % (sb.max(), sbo.max(), time.time()-t0), flush=True)

# grid must span the phantom, including tangential skull rays
gb = np.concatenate([np.arange(0, 8, 1.0), np.arange(8, 24, 2.0)])
gc = np.linspace(0, max(10.0, float(sbo.max()) * 1.05), 11)

cfg_off = dict(P.CFG); cfg_off['ETH'] = ETH; cfg_off['FLUOR'] = 'off'
cfg_on = dict(P.CFG); cfg_on['ETH'] = ETH; cfg_on['FLUOR'] = 'full'

F_off = Forward(cfg=cfg_off, tau_ns=0.0, verbose=False)
F_on = Forward(cfg=cfg_on, tau_ns=0.0, verbose=False)
print('building pile-up grid (%d x %d, MC backend)...' % (len(gb), len(gc)), flush=True)
F_pu = Forward(cfg=cfg_on, tau_ns=A.tau, backend='mc', mc_frames=A.mcframes,
               grid_brain=gb, grid_bone=gc, verbose=True)
print('grid done  %.0fs' % (time.time()-t0), flush=True)

Nl = F_on.Nl
S = ['1 ideal', '2 + charge sharing', '3 + K-escape', '4 + pile-up', '5 + noise']
C = {k: np.zeros((Nl, g.nch, A.nview)) for k in S}
rng = np.random.default_rng(0)

for s in range(0, A.nview, A.chunk):
    e = min(s + A.chunk, A.nview)
    vb, vbo = sb[:, s:e].ravel(), sbo[:, s:e].ravel()
    r = lambda m: m.T.reshape(Nl, g.nch, e - s)
    md_off, _, mi = F_off.mean_cov(vb, vbo)
    md_on, _, _ = F_on.mean_cov(vb, vbo)
    yd, _, md_pu, _ = F_pu.sample(vb, vbo, rng)
    C['1 ideal'][:, :, s:e] = r(mi)
    C['2 + charge sharing'][:, :, s:e] = r(md_off)
    C['3 + K-escape'][:, :, s:e] = r(md_on)
    C['4 + pile-up'][:, :, s:e] = r(md_pu)
    C['5 + noise'][:, :, s:e] = r(yd)
    if s % (A.chunk * 4) == 0:
        print('  views %4d/%d  %.0fs' % (e, A.nview, time.time()-t0), flush=True)

print('\ncounts at a central ray, per stage:')
cc = g.nch // 2
for k in S:
    print('  %-20s %s' % (k, np.array2string(np.round(C[k][:, cc, 0], 1), max_line_width=110)))

show = [0, Nl - 1]
rec = {}
fig, ax = plt.subplots(2 * len(show), len(S), figsize=(3.5 * len(S), 7.0 * len(show)))
for bi, l in enumerate(show):
    for c, k in enumerate(S):
        sino = C[k][l]
        air = sino[5, 0]
        gi = -np.log(np.maximum(sino, 1e-6) / max(air, 1e-9))
        im = ax[2*bi, c].imshow(sino.T, aspect='auto', cmap='gray',
                                norm=LogNorm(max(sino.min(), 1e-2), sino.max()),
                                extent=[0, g.nch, A.nview, 0])
        ax[2*bi, c].set_title('%s\nsinogram, bin %d (%s keV)' % (k, l+1, LBL[l]), fontsize=8)
        ax[2*bi, c].set_xlabel('channel'); ax[2*bi, c].set_ylabel('view')
        img = g.fbp(np.ascontiguousarray(gi)); rec['%s_bin%d' % (k, l+1)] = img
        if c == 0:
            v = np.percentile(img, [1, 99.5])
        ax[2*bi+1, c].imshow(img, cmap='gray', vmin=v[0], vmax=v[1])
        ax[2*bi+1, c].axis('off')
        ax[2*bi+1, c].set_title('reconstruction', fontsize=8)
        print('  panel %s bin %d  %.0fs' % (k, l+1, time.time()-t0), flush=True)
fig.suptitle('Cumulative detector physics (all panels share a display window per row)')
fig.tight_layout(); fig.savefig(FIGS + '/cascade_absolute.png', dpi=125, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for nm, ch in (('air', 20), ('skull edge', cc - 430), ('centre', cc)):
    pass
for j, (nm, ch) in enumerate([('air', 20), ('skull edge', max(cc - 430, 0)), ('centre', cc)]):
    for k in S:
        ax[j].plot(np.arange(Nl), C[k][:, ch, 0], 'o-', ms=4, lw=1.2, label=k)
    ax[j].set_yscale('log'); ax[j].set_title(nm); ax[j].set_xticks(np.arange(Nl))
    ax[j].set_xticklabels(LBL, rotation=45, fontsize=7); ax[j].grid(alpha=.3)
    ax[j].set_xlabel('energy bin (keV)')
ax[0].set_ylabel('counts / pixel / view'); ax[0].legend(fontsize=7)
fig.suptitle('Recorded spectrum as each effect is added')
fig.tight_layout(); fig.savefig(FIGS + '/cascade_spectra.png', dpi=130, bbox_inches='tight')

np.savez_compressed(OUT + '/cascade.npz', **rec,
                    **{('sino_' + k): C[k] for k in S})
print('\nwrote figures/cascade_absolute.png, figures/cascade_spectra.png, outputs/cascade.npz (%.0fs)'
      % (time.time()-t0))

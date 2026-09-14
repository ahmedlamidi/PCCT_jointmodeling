"""Before / after charge sharing, at every level of the data, with the parameters.

Four rows, each the SAME comparison at a different level:

    row 1  the model      what the detector does to one photon
    row 2  the spectrum   what it does to a 120 kVp beam through the head
    row 3  the sinogram   what it does to the measured counts
    row 4  the image      what it does after reconstruction

"before" = ideal detector: same quantum efficiency, same spectrum, perfect
energy assignment, no charge sharing.  "after" = PcTK with FLUOR='off', which
is charge sharing ONLY (no K-escape) -- so the difference between the two
columns is charge sharing and nothing else.

Rows 3-4 are read from outputs/cascade.npz (stage 1 vs stage 2), so this script
is seconds, not hours.  Regenerate that with:  python3 fig_cascade.py --nview 720

Writes the combined sheet AND one PNG per panel:

    figures/physics_before_after_charge_sharing.png
    figures/before_after/00_parameters.png ... 15_image_profile.png

    python3 fig_before_after.py
"""
import argparse
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.gridspec as gs

from paths import FIGS, OUT, PCTK
import pctk_compare as P

_ap = argparse.ArgumentParser()
_ap.add_argument('--bright-is-more', action='store_true',
                 help='matplotlib default sense (high counts / high mu = white). '
                      'Default here is the opposite: DARK = more.')
_A = _ap.parse_args()
# Grayscale sense for the sinogram and reconstruction panels. 'gray_r' puts high
# counts / high mu at BLACK, which is the requested reading. The diverging
# difference panels are untouched -- RdBu_r is already dark at both extremes.
GRAY = 'gray' if _A.bright_is_more else 'gray_r'

ETH = [20., 50., 65., 80.]
LBL = ['20-50', '50-65', '65-80', '80+']
BIN = 0                                   # bin 1 = 20-50 keV, where sharing bites hardest
SUBDIR = os.path.join(FIGS, 'before_after')
os.makedirs(SUBDIR, exist_ok=True)

# ---------------------------------------------------------------- parameters
par = {}
with open(os.path.join(PCTK, '1_inputdata/SRF_param_v32_20170622.csv')) as f:
    for line in f:
        k, v = line.strip().split(',')
        par[k] = v

d = P.load()
cfg_cs = dict(P.CFG); cfg_cs['ETH'] = ETH; cfg_cs['FLUOR'] = 'off'
R_cs, R_id, _ = P.responses(d, cfg_cs)
vEo, vE1 = d['vEo'], d['vE1']
tw = P.binner(d, cfg_cs)
spec = lambda b, bo: d['S'] * np.exp(-d['mu'][:, 0] * b - d['mu'][:, 1] * bo)
cases = [('air', 0., 0.), ('16 cm soft tissue', 16., 0.), ('16 cm soft + 4 cm bone', 16., 4.)]
iE = int(np.where(vE1 == 100)[0][0])
q0 = d['q'][0].reshape(170, 9, 191)
m = vEo >= 6.0
e_pix = (q0[iE][:, m] * vEo[m]).sum(1); e_pix = e_pix / e_pix.sum() * 100

CA = os.path.join(OUT, 'cascade.npz')
if not os.path.exists(CA):
    raise SystemExit('need outputs/cascade.npz -- run: python3 fig_cascade.py --nview 720')
C = np.load(CA)
s_id = C['sino_1 ideal'][BIN]
s_cs = C['sino_2 + charge sharing'][BIN]
s_id2 = C['sino_1 ideal'][1]              # bin 2 = 50-65 keV, the sign-crossing bin
s_cs2 = C['sino_2 + charge sharing'][1]
r_id = C['1 ideal_bin%d' % (BIN + 1)]
r_cs = C['2 + charge sharing_bin%d' % (BIN + 1)]
dif = r_cs - r_id
objm = np.abs(r_id) > np.percentile(np.abs(r_id), 60)
lvl = np.abs(r_id[objm]).mean()
relimg = 100 * dif / lvl
softm = objm & (np.abs(r_id) < np.percentile(np.abs(r_id), 97))
skullm = np.abs(r_id) > np.percentile(np.abs(r_id), 99)
rel = 100 * (s_cs - s_id) / np.maximum(s_id, 1e-9)
cc_ = rel.shape[0] // 2
air_r = 100 * (s_cs[:2].mean() / s_id[:2].mean() - 1)   # ch 0-1 is pure air in every view
ctr_r = 100 * (s_cs[cc_-100:cc_+100].mean() / s_id[cc_-100:cc_+100].mean() - 1)
smax = max(s_id.max(), s_cs.max()); smin = max(min(s_id.min(), s_cs.min()), 1e-1)
vv = np.percentile(r_id, [1, 99.5])

TXT = (
 '$\\bf{Parameters}$   (PcTK 3.2, 1_inputdata/SRF_param_v32_20170622.csv — these are the file\'s values, not assumed)\n\n'
 '$\\bf{Sensor}$        CdTe,  pixel pitch $d_{pix}$ = %s µm,  thickness $d_z$ = %s µm,  density $\\rho$ = %s g/cm³\n'
 '$\\bf{Charge\\ cloud}$  3D Gaussian, radius $r_0$ = %s µm (measured here as $\\sigma\\approx16$ µm),  size independent of energy\n'
 '$\\bf{Electronics}$   noise $\\sigma_e$ = %s keV,  thresholds = %s keV (4 bins)\n'
 '$\\bf{Beam}$          120 kVp, 2.0 mm Al filtration,  $N_0$ = %s photons/pixel/view,  %s views/s\n'
 '$\\bf{This\\ figure}$    FLUOR = \'off\'  → charge sharing only, K-escape disabled.  Geometry: fan beam, R=600 mm, '
 'R$_d$=1080 mm, 1854 ch, 720 views'
) % (par['dpix(um)'], par['dz(um)'], par['rho_PCD(g/cm3)'], par['r_0(um)'], par['sigma_e(keV)'],
     ', '.join('%g' % e for e in ETH), '%g' % P.CFG['N0'], '%g' % P.CFG['VIEWS_PER_SEC'])


# ================================================================== panels
def p_params(ax, fig):
    ax.axis('off')
    ax.text(0.005, 1.02, TXT, va='top', ha='left', fontsize=10.5, family='monospace',
            linespacing=1.6, bbox=dict(boxstyle='round,pad=0.7', fc='#f4f4f8', ec='#888'))


def p_resp_ideal(ax, fig):
    ax.imshow(np.maximum(R_id.T, 1e-8), origin='lower', aspect='auto', cmap='inferno',
              norm=LogNorm(1e-4, R_id.max()), extent=[vE1[0], vE1[-1], vEo[0], vEo[-1]])
    ax.set_title('BEFORE — ideal detector', fontsize=10)
    ax.set_xlabel('incident energy (keV)'); ax.set_ylabel('recorded energy (keV)')


def p_resp_cs(ax, fig):
    im = ax.imshow(np.maximum(R_cs.T, 1e-8), origin='lower', aspect='auto', cmap='inferno',
                   norm=LogNorm(1e-4, R_id.max()), extent=[vE1[0], vE1[-1], vEo[0], vEo[-1]])
    ax.set_title('AFTER — charge sharing', fontsize=10)
    ax.set_xlabel('incident energy (keV)'); ax.set_ylabel('recorded energy (keV)')
    fig.colorbar(im, ax=ax, fraction=.046, pad=.02)


def p_slice(ax, fig):
    ax.semilogy(vEo, np.maximum(R_id[iE], 1e-6), lw=2.0, label='before (ideal)')
    ax.semilogy(vEo, np.maximum(R_cs[iE], 1e-6), lw=1.8, label='after (charge sharing)')
    ax.axvline(100, color='k', ls=':', lw=1.2)
    ax.set_xlim(5, 125); ax.set_ylim(1e-5, 0.5)
    ax.set_xlabel('recorded energy (keV)'); ax.set_ylabel('counts / photon / keV', labelpad=1)
    ax.set_title('one 100 keV photon', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)


def p_3x3(ax, fig):
    imm = ax.imshow(e_pix.reshape(3, 3), cmap='inferno')
    for i in range(3):
        for j in range(3):
            v = e_pix.reshape(3, 3)[i, j]
            ax.text(j, i, '%.1f%%' % v, ha='center', va='center', fontweight='bold',
                    color='w' if v < e_pix.max() * .55 else 'k')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('where that photon\'s energy goes\n%.0f%% lands in pixels it never hit'
                 % (100 - e_pix.reshape(3, 3)[1, 1]), fontsize=10)
    fig.colorbar(imm, ax=ax, fraction=.046, pad=.02).set_label('% of energy', fontsize=8)


def _bars(ax, fig, c):
    nm, b, bo = cases[c]
    s = spec(b, bo)
    wi = P.CFG['N0'] * (tw(R_id).T @ s); wm = P.CFG['N0'] * (tw(R_cs).T @ s)
    x = np.arange(4); w = .38
    ax.bar(x - w/2, wi, w, label='before (ideal)', color='C0')
    ax.bar(x + w/2, wm, w, label='after (sharing)', color='C3')
    for k in range(4):
        pc = 100 * (wm[k] - wi[k]) / max(wi[k], 1e-9)
        ax.text(k, max(wi[k], wm[k]) * 1.12, '%+.0f%%' % pc, ha='center', fontsize=9,
                color='C3' if pc > 0 else 'C0', fontweight='bold')
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(LBL)
    ax.set_ylim(top=max(wi.max(), wm.max()) * 3.0)
    ax.set_xlabel('energy bin (keV)'); ax.set_title('%s' % nm, fontsize=10)
    ax.set_ylabel('counts / pixel / view'); ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=8)


def p_bars_air(ax, fig):  _bars(ax, fig, 0)
def p_bars_soft(ax, fig): _bars(ax, fig, 1)
def p_bars_bone(ax, fig): _bars(ax, fig, 2)


def p_binshift(ax, fig):
    for nm, b, bo in cases:
        s = spec(b, bo)
        wi = tw(R_id).T @ s; wm = tw(R_cs).T @ s
        ax.plot(np.arange(4), 100 * (wm - wi) / np.maximum(wi, 1e-12), 'o-', lw=1.8, label=nm)
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(LBL)
    ax.set_xlabel('energy bin (keV)'); ax.set_ylabel('change in counts (%)')
    ax.set_title('change in counts per bin', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)


def _sino(ax, fig, S, t):
    im = ax.imshow(S.T, aspect='auto', cmap=GRAY, norm=LogNorm(smin, smax),
                   extent=[0, S.shape[0], S.shape[1], 0])
    ax.set_title('%s\nsinogram, bin 1 (20-50 keV)' % t, fontsize=10)
    ax.set_xlabel('channel'); ax.set_ylabel('view')
    fig.colorbar(im, ax=ax, fraction=.046, pad=.02).set_label('counts', fontsize=8)


def p_sino_ideal(ax, fig): _sino(ax, fig, s_id, 'BEFORE — ideal')
def p_sino_cs(ax, fig):    _sino(ax, fig, s_cs, 'AFTER — charge sharing')


def p_sino_diff(ax, fig):
    im = ax.imshow(rel.T, aspect='auto', cmap='RdBu_r',
                   norm=SymLogNorm(linthresh=10, vmin=-100, vmax=3000, base=10),
                   extent=[0, rel.shape[0], rel.shape[1], 0])
    ax.set_title('DIFFERENCE  (after - before)/before\nair %+.0f%%, through the head %+.0f%%'
                 % (air_r, ctr_r), fontsize=10)
    ax.set_xlabel('channel'); ax.set_ylabel('view')
    fig.colorbar(im, ax=ax, fraction=.046, pad=.02).set_label('% change', fontsize=8)


V0 = s_id.shape[1] // 2


def p_sino_profile(ax, fig):
    ax.semilogy(s_id[:, V0], lw=1.6, color='C0', label='before (ideal)')
    ax.semilogy(s_cs[:, V0], lw=1.4, color='C3', label='after (sharing)')
    ax.set_xlabel('channel'); ax.set_ylabel('counts, bin 1')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=.3)
    r2 = s_cs[:, V0] / np.maximum(s_id[:, V0], 1e-9)
    ax2 = ax.twinx()
    ax2.semilogy(r2, lw=1.3, color='0.25', ls=':')
    ax2.axhline(1.0, color='k', lw=.8, alpha=.5)
    ax2.set_ylabel('ratio after/before (dotted)', fontsize=8, color='0.25')
    ax2.tick_params(axis='y', labelsize=7, colors='0.25')
    ax.set_title('one view through the head (view %d)\nratio %.1fx in air, up to %.0fx'
                 % (V0, r2[0], r2[s_id[:, V0] < 0.5 * s_id[:2].mean()].max()), fontsize=9)


def p_sino_ratio_bins(ax, fig):
    for L, lb in enumerate(LBL):
        a = C['sino_1 ideal'][L][:, V0]; b = C['sino_2 + charge sharing'][L][:, V0]
        ax.semilogy(b / np.maximum(a, 1e-9), lw=1.6, label='bin %d (%s keV)' % (L + 1, lb))
    ax.axhline(1.0, color='k', lw=1.2)
    ax.set_xlabel('channel'); ax.set_ylabel('counts after / counts before')
    ax.set_title('ratio per bin, view %d' % V0, fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)


def p_sino_profile_bin2(ax, fig):
    a, b = s_id2[:, V0], s_cs2[:, V0]
    ax.semilogy(a, lw=1.6, color='C0', label='before (ideal)')
    ax.semilogy(b, lw=1.4, color='C3', label='after (sharing)')
    ax.set_xlabel('channel'); ax.set_ylabel('counts, bin 2 (50-65 keV)')
    obj2 = a < 0.5 * a[:2].mean()
    ax.set_title('bin 2 (50-65 keV)\nair %+.0f%%, object %+.0f%% to %+.0f%%'
                 % (100 * (b[0] / a[0] - 1),
                    100 * ((b[obj2] / a[obj2]).min() - 1),
                    100 * ((b[obj2] / a[obj2]).max() - 1)), fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=.3)


def _img(ax, fig, I, t):
    ax.imshow(I, cmap=GRAY, vmin=vv[0], vmax=vv[1]); ax.axis('off')
    ax.set_title('%s\nreconstruction, bin 1' % t, fontsize=10)


def p_img_ideal(ax, fig): _img(ax, fig, r_id, 'BEFORE — ideal')
def p_img_cs(ax, fig):    _img(ax, fig, r_cs, 'AFTER — charge sharing')


def p_img_diff(ax, fig):
    lim = np.percentile(np.abs(relimg[objm]), 99)
    im = ax.imshow(relimg, cmap='RdBu_r', vmin=-lim, vmax=lim); ax.axis('off')
    ax.set_title('DIFFERENCE, %% of mean object level\nskull %+.0f%%, soft tissue %+.0f%%'
                 % (100 * dif[skullm].mean() / lvl, 100 * dif[softm].mean() / lvl), fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.02).set_label('% of mean object $\\mu$', fontsize=8)


def p_img_profile(ax, fig):
    mid = r_id.shape[0] // 2
    ax.plot(r_id[mid], lw=1.6, label='before (ideal)')
    ax.plot(r_cs[mid], lw=1.4, label='after (sharing)')
    ax.set_xlabel('pixel'); ax.set_ylabel('$\\mu$ (a.u.)')
    ax.set_title('central horizontal profile', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)


PANELS = [
    ('00_parameters',        p_params,        (14, 3.0)),
    ('01_response_ideal',    p_resp_ideal,    (5.6, 4.8)),
    ('02_response_sharing',  p_resp_cs,       (6.4, 4.8)),
    ('03_photon_100keV',     p_slice,         (6.0, 4.8)),
    ('04_energy_3x3',        p_3x3,           (6.0, 4.8)),
    ('05_bins_air',          p_bars_air,      (5.6, 4.6)),
    ('06_bins_soft',         p_bars_soft,     (5.6, 4.6)),
    ('07_bins_bone',         p_bars_bone,     (5.6, 4.6)),
    ('08_bin_shift',         p_binshift,      (6.2, 4.6)),
    ('09_sinogram_ideal',    p_sino_ideal,    (6.0, 5.0)),
    ('10_sinogram_sharing',  p_sino_cs,       (6.0, 5.0)),
    ('11_sinogram_diff',     p_sino_diff,     (6.2, 5.0)),
    ('12_sinogram_profile',  p_sino_profile,  (6.0, 5.0)),
    ('13_image_ideal',       p_img_ideal,     (5.4, 5.4)),
    ('14_image_sharing',     p_img_cs,        (5.4, 5.4)),
    ('15_image_diff',        p_img_diff,      (6.0, 5.4)),
    ('16_image_profile',     p_img_profile,   (6.0, 5.0)),
]

EXTRA = [
    ('17_sinogram_ratio_bins',   p_sino_ratio_bins,   (6.6, 5.0)),
    ('18_sinogram_profile_bin2', p_sino_profile_bin2, (6.2, 5.0)),
]

# ------------------------------------------------------------ combined sheet
plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(19, 22))
G = gs.GridSpec(5, 4, height_ratios=[0.60, 1, 1, 1, 1], hspace=.52, wspace=.42)
slots = [G[0, :]] + [G[r, c] for r in range(1, 5) for c in range(4)]
for (nm, fn, _), sl in zip(PANELS, slots):
    fn(fig.add_subplot(sl), fig)
fig.suptitle('Charge sharing: model, spectrum, sinogram, image', fontsize=15, y=.998)
out = FIGS + '/physics_before_after_charge_sharing.png'
fig.savefig(out, dpi=110, bbox_inches='tight'); plt.close(fig)
print('wrote', out)

# ------------------------------------------------------------ one PNG each
for nm, fn, size in PANELS + EXTRA:
    f1 = plt.figure(figsize=size)
    fn(f1.add_subplot(111), f1)
    f1.tight_layout()
    fp = os.path.join(SUBDIR, nm + '.png')
    f1.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(f1)
    print('  ', fp)

# ------------------------------------------------------------------ numbers
print('\nBIN COUNTS  (counts/pixel/view, N0=%g)' % P.CFG['N0'])
print('%-26s %10s %10s %10s %10s' % ('path', *LBL))
for nm, b, bo in cases:
    s = spec(b, bo)
    wi = P.CFG['N0'] * (tw(R_id).T @ s); wm = P.CFG['N0'] * (tw(R_cs).T @ s)
    print('  %-22s before %10.1f %10.1f %10.1f %10.1f' % (nm, *wi))
    print('  %-22s after  %10.1f %10.1f %10.1f %10.1f' % ('', *wm))
    print('  %-22s change %9.1f%% %9.1f%% %9.1f%% %9.1f%%'
          % ('', *(100 * (wm - wi) / np.maximum(wi, 1e-9))))
print('\nSINOGRAM bin 1   air %+.1f%%   through head %+.1f%%' % (air_r, ctr_r))
print('IMAGE    bin 1   skull %+.1f%%   soft tissue %+.1f%%  (of mean object level)'
      % (100 * dif[skullm].mean() / lvl, 100 * dif[softm].mean() / lvl))

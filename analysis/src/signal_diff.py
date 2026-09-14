"""Signal at every stage of the pipeline, and the difference each stage makes.

    1 ideal            perfect spectrometer, same QE
    2 + charge sharing PcTK, fluorescence off
    3 + K-escape       PcTK full
    4 + pile-up        rate-dependent stage (MC backend)

Prints, per operating point: counts at each stage, the DELTA each stage adds,
and the running total error against ideal. Then a figure.
"""
import argparse, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import FIGS, OUT
import pctk_compare as P
from pipeline import Forward

ap = argparse.ArgumentParser()
ap.add_argument('--tau', type=float, default=30.0)
ap.add_argument('--bins', type=int, default=9, choices=[4, 9])
A = ap.parse_args()

ETH = ([20., 50., 65., 80.] if A.bins == 4
       else [20., 30., 40., 50., 60., 70., 80., 90., 100.])
LBL = ['%g-%s' % (ETH[i], ('%g' % ETH[i+1]) if i < len(ETH)-1 else 'inf')
       for i in range(len(ETH))]
PTS = [('air', 0., 0.), ('5 cm soft', 5., 0.), ('15 cm soft', 15., 0.),
       ('20 cm soft', 20., 0.), ('20 cm + 2 cm bone', 20., 2.),
       ('20 cm + 6 cm bone', 20., 6.)]

t0 = time.time()
cfg_off = dict(P.CFG); cfg_off['ETH'] = ETH; cfg_off['FLUOR'] = 'off'
cfg_on = dict(P.CFG); cfg_on['ETH'] = ETH; cfg_on['FLUOR'] = 'full'
F_off = Forward(cfg=cfg_off, tau_ns=0.0, verbose=False)
F_on = Forward(cfg=cfg_on, tau_ns=0.0, verbose=False)
gb = np.concatenate([np.arange(0, 8, 1.), np.arange(8, 24, 2.)])
gc = np.linspace(0, 8, 9)
import os
gcache = OUT + '/pugrid_tau%g_bins%d.npz' % (A.tau, len(ETH))
F_pu = Forward.__new__(Forward)
if os.path.exists(gcache):
    print('loading cached pile-up grid %s' % os.path.basename(gcache), flush=True)
    F_pu = Forward(cfg=cfg_on, tau_ns=0.0, verbose=False)
    z = np.load(gcache)
    F_pu.grid = dict(brain=z['brain'], bone=z['bone'], mean=z['mean'], cov=z['cov'])
else:
    print('building pile-up grid (tau=%g ns), ~8 min...' % A.tau, flush=True)
    F_pu = Forward(cfg=cfg_on, tau_ns=A.tau, backend='mc', mc_frames=150,
                   grid_brain=gb, grid_bone=gc, verbose=False)
    np.savez_compressed(gcache, **F_pu.grid)
    print('  cached to %s' % os.path.basename(gcache), flush=True)
print('  ready %.0fs\n' % (time.time()-t0), flush=True)

S = ['1 ideal', '2 +sharing', '3 +K-escape', '4 +pile-up']
res = {}
for nm, a_, b_ in PTS:
    va, vb = np.array([a_]), np.array([b_])
    m_off, _, m_id = F_off.mean_cov(va, vb)
    m_on, _, _ = F_on.mean_cov(va, vb)
    m_pu, _, _ = F_pu.mean_cov(va, vb)
    res[nm] = np.stack([m_id[0], m_off[0], m_on[0], m_pu[0]])

np.savez(OUT + '/signal_diff.npz', **{k: v for k, v in res.items()}, eth=np.array(ETH))
w = max(len(l) for l in LBL) + 1

for nm, a_, b_ in PTS:
    R = res[nm]
    print('=' * 108)
    print('%s   (soft %g cm, bone %g cm)   total counts: ideal %.0f -> final %.0f'
          % (nm, a_, b_, R[0].sum(), R[3].sum()))
    print('  %-13s' % 'bin (keV)' + ''.join('%*s' % (w+2, l) for l in LBL))
    for i, s in enumerate(S):
        print('  %-13s' % s + ''.join('%*.1f' % (w+2, v) for v in R[i]))
    print('  ' + '-' * 104)
    for i, s in enumerate(S[1:], 1):
        d = R[i] - R[i-1]
        print('  %-13s' % ('D ' + s.split('+')[1]) + ''.join('%+*.1f' % (w+2, v) for v in d))
    err = 100 * (R[3] - R[0]) / np.maximum(R[0], 1e-9)
    print('  %-13s' % 'total err %' + ''.join('%+*.0f' % (w+2, v) for v in err))
    print()

fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
for k, (nm, a_, b_) in enumerate(PTS):
    A_ = ax[k // 3, k % 3]; R = res[nm]
    for i, s in enumerate(S):
        A_.plot(np.arange(len(ETH)), R[i], 'o-', ms=4, lw=1.4, label=s)
    A_.set_yscale('log'); A_.set_title(nm, fontsize=10); A_.grid(alpha=.3)
    A_.set_xticks(np.arange(len(ETH)))
    A_.set_xticklabels(LBL, rotation=45, fontsize=6)
    if k == 0: A_.legend(fontsize=8)
    if k % 3 == 0: A_.set_ylabel('counts / pixel / view')
fig.suptitle('Signal at each pipeline stage (tau = %g ns, %d bins)' % (A.tau, len(ETH)))
fig.tight_layout(); fig.savefig(FIGS + '/signal_by_stage.png', dpi=130, bbox_inches='tight')

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for j, (i, s) in enumerate([(1, 'charge sharing'), (2, 'K-escape'), (3, 'pile-up')]):
    for nm, a_, b_ in PTS:
        R = res[nm]
        d = 100 * (R[i] - R[i-1]) / np.maximum(R[i-1], 1e-9)
        ax[j].plot(np.arange(len(ETH)), d, 'o-', ms=4, lw=1.3, label=nm)
    ax[j].axhline(0, color='k', lw=.8); ax[j].set_title('change from %s (%%)' % s, fontsize=10)
    ax[j].set_xticks(np.arange(len(ETH)))
    ax[j].set_xticklabels(LBL, rotation=45, fontsize=6); ax[j].grid(alpha=.3)
    ax[j].set_xlabel('energy bin (keV)')
ax[0].set_ylabel('% change vs previous stage'); ax[0].legend(fontsize=7)
fig.suptitle('What each effect adds, as a percentage of the stage before it')
fig.tight_layout(); fig.savefig(FIGS + '/signal_delta_per_stage.png', dpi=130, bbox_inches='tight')
print('wrote figures/signal_by_stage.png, figures/signal_delta_per_stage.png  (%.0fs)' % (time.time()-t0))

"""One figure: is pulse pile-up present? Three independent signatures.

A  counts appear ABOVE the tube voltage -- two photons recorded as one.
   No single photon from a 120 kVp source can deposit more than 120 keV.
B  the count-rate curve bends over and saturates instead of rising linearly.
C  the counts go SUB-Poisson (Fano < 1), which matters for any uncertainty work.
"""
import argparse, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '.')
from paths import FIGS
import pctk_compare as P
import pileup_mc_ref as MR

ap = argparse.ArgumentParser()
ap.add_argument('--tau', type=float, default=30.0, help='dead time, ns')
ap.add_argument('--t_p', type=float, default=10.0)
ap.add_argument('--T', type=float, default=25.0)
ap.add_argument('--kvp', type=float, default=120.0)
A = ap.parse_args()

d = P.load(); cfg = dict(P.CFG)
R, _, _ = P.responses(d, cfg)
vEo = d['vEo']; Et = cfg['TRIGGER_KEV']
tau, t_p, T = A.tau*1e-9, A.t_p*1e-9, A.T*1e-9
VPS = 4000.0; frame_T = 1.0 / VPS
dep = R.T @ d['S']                                   # unattenuated deposited spectrum
per_photon = dep[vEo >= Et].sum()
Emax = vEo[dep > dep.max()*1e-6].max()               # highest energy one photon can deposit

rng = np.random.default_rng(0)
lts = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0])
rates = lts / tau
out = []
spec_show = {}
t0 = time.time()
for lt, rate in zip(lts, rates):
    nf = max(120, int(min(600, 2e6 / max(rate*frame_T, 1))))
    c, recs = MR.mc_pulse_train(dep, vEo, rate, tau, t_p, T, Et,
                                cfg['ETH'], nf, frame_T, rng, max_counts=3_000_000)
    Ttot = len(c) * frame_T
    out.append((lt, rate, len(recs)/Ttot, c.sum(1).mean(), c.sum(1).var()))
    if lt in (0.02, 0.4, 1.0, 3.0):
        spec_show[lt] = recs
    print('  lam*tau=%.2f  %d frames  %.0fs' % (lt, len(c), time.time()-t0), flush=True)
out = np.array(out)

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))

# ---- A: spectrum beyond the tube voltage -------------------------------
bins = np.arange(0, 320, 2.0)
for lt in sorted(spec_show):
    h, _ = np.histogram(spec_show[lt], bins=bins, density=True)
    ax[0].semilogy(bins[:-1], np.maximum(h, 1e-7), lw=1.9,
                   label=r'$\lambda\tau$ = %.2f' % lt)
ax[0].axvline(A.kvp, color='k', lw=2.4)
ax[0].axvspan(A.kvp, bins[-1], color='crimson', alpha=.12)
ax[0].text(A.kvp+6, 3e-2, 'IMPOSSIBLE for one photon\n(tube voltage %.0f kVp)' % A.kvp,
           fontsize=10, color='crimson')
ax[0].set_xlim(0, 300); ax[0].set_ylim(1e-6, 1e-1)
ax[0].set_xlabel('recorded pulse height (keV)'); ax[0].set_ylabel('probability density')
ax[0].set_title('A.  Counts appear above the tube voltage.\nTwo photons recorded as one.',
                fontsize=12)
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

# ---- B: count-rate curve ----------------------------------------------
inc = out[:, 1]                       # rate passed to the MC IS the triggerable rate
ax[1].plot(inc/1e6, inc/1e6, 'k--', lw=1.8, label='ideal detector (no losses)')
ax[1].plot(inc/1e6, out[:, 2]/1e6, 'o-', color='C3', lw=2.2, ms=6, label='simulated')
ax[1].plot(inc/1e6, (inc/(1 + inc*tau))/1e6, ':', color='C0', lw=2.2,
           label=r'non-paralyzable  $m=n/(1+n\tau)$')
ax[1].set_xlabel('incident rate (Mcps / pixel)'); ax[1].set_ylabel('recorded rate (Mcps / pixel)')
ax[1].set_title('B.  Output saturates instead of\nrising with input', fontsize=12)
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)

# ---- C: Fano ------------------------------------------------------------
fano = out[:, 4] / np.maximum(out[:, 3], 1e-9)
ax[2].plot(out[:, 0], fano, 'o-', color='C2', lw=2.2, ms=6)
ax[2].axhline(1.0, color='k', lw=2.0, ls='--', label='Poisson (variance = mean)')
ax[2].fill_between(out[:, 0], 0, 1, color='C2', alpha=.10)
ax[2].text(1.2, 0.55, 'sub-Poisson:\nless random than Poisson', fontsize=10, color='C2')
ax[2].set_xlabel(r'relative count rate  $\lambda\tau$')
ax[2].set_ylabel('Fano factor  (variance / mean)')
ax[2].set_title('C.  The statistics change too,\nnot just the mean', fontsize=12)
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3); ax[2].set_ylim(0, 1.25)

fig.tight_layout()
fig.savefig(FIGS + '/physics_pileup_evidence.png', dpi=140, bbox_inches='tight')

print('\nhighest energy a single photon can deposit: %.0f keV' % Emax)
print('%-9s %-14s %-14s %-10s %s' % ('lam*tau', 'incident Mcps', 'recorded Mcps', 'loss %', 'Fano'))
for (lt, r, rec, mu, va), f in zip(out, fano):
    i = r
    print('%-9.2f %-14.2f %-14.2f %-10.1f %.3f' % (lt, i/1e6, rec/1e6, 100*(1-rec/i), f))
frac = np.mean(spec_show[3.0] > A.kvp) * 100
print('\nat lam*tau=3.0, %.1f%% of recorded pulses exceed the tube voltage' % frac)
print('wrote figures/physics_pileup_evidence.png')

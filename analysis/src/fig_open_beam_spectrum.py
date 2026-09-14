"""Spectral profile in an open beam area -- reproduction of the reference figure.

Four curves, same labels as the paper:
    Sino   what a perfect detector records: the incident spectrum itself
    CS     after charge sharing + K-escape
    CSPN   CS plus Poisson noise
    PU     CS plus pulse pile-up

Open beam = no object in the way, so this is the unattenuated spectrum.
PcTK's shipped 120 kVp source has the tungsten characteristic lines
(local maxima at 39, 60 and 68 keV), so the red curve is directly comparable.
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '.')
from paths import FIGS
import pctk_compare as P
import pileup_mc_ref as MR

ap = argparse.ArgumentParser()
ap.add_argument('--lamtau', type=float, default=0.4, help='relative count rate for the PU curve')
ap.add_argument('--tau', type=float, default=30.0)
ap.add_argument('--t_p', type=float, default=10.0)
ap.add_argument('--T', type=float, default=25.0)
ap.add_argument('--scale', type=float, default=None,
                help='display scale; default puts the Sino peak near 1.6e4 as in the paper')
A = ap.parse_args()

d = P.load(); cfg = dict(P.CFG)
vEo, vE1, S = d['vEo'], d['vE1'], d['S']
R, _, _ = P.responses(d, cfg)          # (170 E_in, 191 E_out), summed over the 3x3
Et = cfg['TRIGGER_KEV']
N0 = cfg['N0']

# --- Sino: perfect detector. Counts land exactly at the incident energy.
sino = np.zeros_like(vEo)
for i, E in enumerate(vE1):
    sino[int(np.argmin(np.abs(vEo - E)))] += N0 * S[i]

# --- CS: charge sharing + K-escape
cs = N0 * (R.T @ S)

# --- CSPN: CS with Poisson noise
rng = np.random.default_rng(0)
cspn = rng.poisson(np.maximum(cs, 0)).astype(float)

# --- PU: CS then pile-up, at the requested relative count rate
dep = R.T @ S
tau, t_p, T = A.tau*1e-9, A.t_p*1e-9, A.T*1e-9
rate = A.lamtau / tau
frame_T = 1.0 / 4000.0
_, recs = MR.mc_pulse_train(dep, vEo, rate, tau, t_p, T, Et, cfg['ETH'],
                            400, frame_T, rng, max_counts=3_000_000)
h, _ = np.histogram(recs, bins=np.arange(-0.5, 191.5, 1.0))
pu = h / h.sum() * cs[vEo >= Et].sum()          # normalise to the same total counts

sc = A.scale if A.scale else 1.6e4 / max(sino.max(), 1e-9)
plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(8.2, 5.4))
m = (vEo >= 10) & (vEo <= 125)
ax.plot(vEo[m], sc*sino[m], '-',   color='red',       lw=2.0, label='Sino')
ax.plot(vEo[m], sc*cs[m],   '-',   color='limegreen', lw=2.4, label='CS')
ax.plot(vEo[m], sc*cspn[m], '--',  color='navy',      lw=1.8, label='CSPN')
ax.plot(vEo[m], sc*pu[m],   '-.',  color='deepskyblue', lw=1.8, label='PU')
ax.set_xlabel('Energy Window/keV'); ax.set_ylabel('Counts')
ax.set_xlim(10, 125); ax.set_ylim(0, None)
ax.legend(fontsize=12, frameon=True)
fig.tight_layout()
fig.savefig(FIGS + '/physics_open_beam_spectrum.png', dpi=140, bbox_inches='tight')

def at(c, e): return sc*c[int(np.argmin(np.abs(vEo-e)))]
print('counts at selected energies (display-scaled, same factor for all curves):')
print('  %-6s %10s %10s %10s %10s' % ('keV', 'Sino', 'CS', 'CSPN', 'PU'))
for e in (10, 15, 20, 30, 40, 60, 68, 80, 100, 120):
    print('  %-6d %10.0f %10.0f %10.0f %10.0f' % (e, at(sino,e), at(cs,e), at(cspn,e), at(pu,e)))
print('\nSino peak %.0f at %g keV; CS at 10 keV is %.1fx the Sino peak'
      % (sc*sino.max(), vEo[sino.argmax()], at(cs,10)/(sc*sino.max())))
print('wrote figures/physics_open_beam_spectrum.png')

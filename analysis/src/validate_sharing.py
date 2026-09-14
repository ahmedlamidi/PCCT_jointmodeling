"""Independent physics checks on PcTK's charge-sharing / fluorescence model.

Every check compares PcTK against something OUTSIDE PcTK -- textbook atomic
energies, or pure geometry -- rather than against its own tables.

  1  K-edge positions          vs CdTe atomic data (Cd 26.71, Te 31.81 keV)
  2  fluorescence emission     vs Cd/Te K-alpha (23.17 / 27.47 keV)
  3  escape peak position      vs E1 - K-alpha
  4  energy conservation       deposited energy vs incident x QE
  5  shared-charge fraction    vs a geometry-only Monte Carlo of a charge cloud
                               of radius r0 landing uniformly in a dpix pixel
"""
import numpy as np, h5py, sys
sys.path.insert(0, '.')
from paths import PCTK, CACHE
import pctk_compare as P

CD_EDGE, TE_EDGE = 26.71, 31.81          # K absorption edges, keV
CD_KA,  TE_KA = 23.17, 27.47             # K-alpha1 emission, keV
R0, DPIX = 24.0, 225.0                   # from SRF_param csv, micron

d = P.load(); vEo, vE1, Pr = d['vEo'], d['vE1'], d['Pr']
diag = d['diagE'].reshape(170, 9, 191)

print('=' * 78)
print('1. K-EDGE POSITIONS   where the no-fluorescence branch Pr(q0) steps down')
q0 = Pr[0]
drop = np.where(np.abs(np.diff(q0)) > 0.02)[0]
for i in drop[:4]:
    print('   step between %.0f and %.0f keV   Pr(q0): %.4f -> %.4f'
          % (vE1[i], vE1[i+1], q0[i], q0[i+1]))
print('   textbook: Cd K-edge %.2f keV, Te K-edge %.2f keV' % (CD_EDGE, TE_EDGE))

print('=' * 78)
print('2/3. FLUORESCENCE AND ESCAPE PEAKS, centre pixel')
print('   %-7s %-28s %-28s' % ('E1', 'fluorescence line (keV)', 'escape peak (keV)'))
for E in (40, 60, 80, 100, 120):
    i = int(np.where(vE1 == E)[0][0])
    s = diag[i, 4]
    lo = s.copy(); lo[(vEo < 15) | (vEo > 35)] = 0          # reabsorbed fluorescence
    hi = s.copy(); hi[(vEo < E - 40) | (vEo > E - 8)] = 0   # escape region
    print('   %-7d %-28s %-28s'
          % (E, '%.1f  (Cd %.2f / Te %.2f)' % (vEo[lo.argmax()], CD_KA, TE_KA),
             '%.1f  (expect %.1f-%.1f)' % (vEo[hi.argmax()], E - TE_KA, E - CD_KA)))

print('=' * 78)
print('4. ENERGY CONSERVATION   sum(E_out x counts) over 9 pixels vs E1 x QE')
QE = Pr.sum(0)
print('   %-6s %10s %12s %12s %8s' % ('E1', 'QE', 'deposited', 'E1*QE', 'ratio'))
for E in (30, 50, 80, 100, 140):
    i = int(np.where(vE1 == E)[0][0])
    dep = (diag[i].sum(0) * vEo).sum()
    exp = E * QE[i]
    print('   %-6d %10.4f %12.2f %12.2f %8.4f' % (E, QE[i], dep, exp, dep / exp))
print('   ratio < 1 is expected: escaped fluorescence leaves the detector.')

print('=' * 78)
print('5. SHARED-CHARGE FRACTION   PcTK vs geometry alone')
rng = np.random.default_rng(0)
n = 400_000
cx, cy = rng.uniform(0, DPIX, n), rng.uniform(0, DPIX, n)      # cloud centres
ang = rng.uniform(0, 2*np.pi, n); rad = R0*np.sqrt(rng.uniform(0, 1, n))
px, py = cx + rad*np.cos(ang), cy + rad*np.sin(ang)            # charge elements
outside = ((px < 0) | (px > DPIX) | (py < 0) | (py > DPIX)).mean()
straddle = (((cx < R0) | (cx > DPIX-R0) | (cy < R0) | (cy > DPIX-R0))).mean()
print('   geometry only (uniform disc r0=%.0f um in a %.0f um pixel):' % (R0, DPIX))
print('     charge landing outside the struck pixel : %.4f' % outside)
print('     events whose cloud straddles a boundary : %.4f' % straddle)
for E in (40, 60, 100):
    i = int(np.where(vE1 == E)[0][0])
    e_pix = (diag[i] * vEo).sum(1)
    frac = 1.0 - e_pix[4] / e_pix.sum()
    print('   PcTK at E1=%3d keV: energy fraction outside struck pixel = %.4f' % (E, frac))

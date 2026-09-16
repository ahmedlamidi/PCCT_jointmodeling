"""NIST ICRU-44 attenuation for the materials in Morovati et al. 2025's phantoms.

Source: Hubbell & Seltzer, NIST Standard Reference Database 126, X-Ray Mass
Attenuation Coefficients, https://physics.nist.gov/PhysRefData/XrayMassCoef/
    Table 2 -> densities (g/cm^3);  Table 4 -> mu/rho per material (ComTab pages).

Morovati et al. (doi 10.1088/1361-6560/adaf71) list "soft tissue, adipose tissue,
brain gray matter, white matter, blood, and cortical bone", with water filling the
gaps. ICRU-44, and so NIST, has ONE entry for grey and white matter together, so
both map to 'brain' -- in this arm they are the same material. Stated, not fixed.

Two-basis representation (Alvarez & Macovski 1976, doi 10.1088/0031-9155/21/5/002):
each material's mu(E) is fitted as a*mu_brain(E) + b*mu_bone(E) over 20-120 keV.
With this all-NIST basis every material fits to <1% max error (adipose 0.84% is the
worst; blood 0.08%, soft tissue 0.02%, water 0.18%). So a ray through any mix of
them is still two numbers -- brain-equivalent cm and bone-equivalent cm -- and the
pile-up grid, normalisation and 2-D manifold carry over unchanged. b can be
NEGATIVE (adipose -0.041, water -0.004), so the bone axis of the pile-up grid must
start below zero.

Why not PcTK's own basis (m2_mukE.csv): its brain differs from NIST brain by up to
6.5% (at 24 keV), so fitting NIST tissues to it leaves ~3.3% max error. That is a
database mismatch, not a failure of the two-basis model.

Between NIST's tabulated energies (10 keV apart at 20-60 keV, 20 keV apart at
60-100 keV) mu/rho is interpolated linearly in log-log -- my choice, not NIST's
prescription. Every material is interpolated the same way.

    python3 nist_materials.py --build    # fetch NIST, rewrite nist_icru44_mu.csv (needs internet, once)
    python3 nist_materials.py            # print densities and the two-basis fit
"""
import argparse, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'nist_icru44_mu.csv')
URL = 'https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/%s.html'
# name -> (NIST ComTab page, density g/cm^3 from NIST Table 2)
NIST = dict(adipose=('adipose', 0.950), blood=('blood', 1.060), bone=('bone', 1.920),
            brain=('brain', 1.040), soft_tissue=('tissue', 1.060), water=('water', 1.000))
NAMES = list(NIST)
E_KEV = np.arange(1, 171, dtype=float)   # PcTK's grid: pctk_compare.load() keeps rows 1-170 keV


def _parse(html):
    """NIST ComTab page -> (N, 2) [energy keV, mu/rho cm^2/g]."""
    t = re.sub(r'<[^>]+>', ' ', html).replace('&nbsp;', ' ')
    sci = r'([0-9]\.[0-9]+E[-+][0-9]+)'
    rows = np.array([[float(e) * 1e3, float(m)] for e, m, _ in
                     re.findall(sci + r'\s+' + sci + r'\s+' + sci, t)])
    # each page repeats its table in a second format; keep the first copy only
    back = np.nonzero(np.diff(rows[:, 0]) < 0)[0]
    return rows[:back[0] + 1] if len(back) else rows


def _loglog(e_kev, mu_rho, E):
    e = e_kev.copy()
    # an absorption edge is a repeated energy; nudge the below-edge entry down so
    # an energy exactly at the edge takes the above-edge value
    for i in range(1, len(e)):
        if e[i] == e[i - 1]:
            e[i - 1] *= 1 - 1e-9
    return np.exp(np.interp(np.log(E), np.log(e), np.log(mu_rho)))


def build():
    import urllib.request
    cols = []
    for name in NAMES:
        page, rho = NIST[name]
        # NIST answers 403 to Python's default User-Agent
        req = urllib.request.Request(URL % page, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=60).read().decode('latin-1')
        tb = _parse(html)
        cols.append(rho * _loglog(tb[:, 0], tb[:, 1], E_KEV))
        print('  %-12s %3d NIST energies  mu(60 keV) = %.4f /cm' % (name, len(tb), cols[-1][59]))
    hdr = 'E(keV),' + ','.join('mu_%s(cm^-1)' % n for n in NAMES)
    np.savetxt(CSV, np.column_stack([E_KEV] + cols), delimiter=',', header=hdr,
               comments='', fmt='%.6g')
    print('wrote', CSV)


def load():
    """-> dict name -> linear attenuation mu (170,), cm^-1, on E_KEV."""
    tb = np.loadtxt(CSV, delimiter=',', skiprows=1)
    return {n: tb[:, i + 1] for i, n in enumerate(NAMES)}


def basis():
    """(170, 2) [NIST brain, NIST cortical bone] -- drop-in for m2_mukE.csv's columns."""
    mu = load()
    return np.column_stack([mu['brain'], mu['bone']])


def two_basis(lo=20.0, hi=120.0):
    """-> dict name -> (a, b, max relative error %): mu ~ a*mu_brain + b*mu_bone on [lo, hi] keV.

    Relative least squares (each energy weighted by 1/mu), so low-energy points,
    where mu is large, do not dominate the fit.
    """
    mu = load()
    sel = (E_KEV >= lo) & (E_KEV <= hi)
    A = np.column_stack([mu['brain'][sel], mu['bone'][sel]])
    out = {}
    for n in NAMES:
        y = mu[n][sel]
        w = 1.0 / y
        c = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        out[n] = (float(c[0]), float(c[1]), float(np.abs(A @ c / y - 1).max() * 100))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true', help='fetch NIST and rewrite the CSV')
    if ap.parse_args().build:
        build()
    print('%-12s %8s %8s %8s %10s' % ('material', 'rho', 'a', 'b', 'max err %'))
    for n, (a_, b_, err) in two_basis().items():
        print('%-12s %8.3f %8.3f %+8.3f %10.2f' % (n, NIST[n][1], a_, b_, err))

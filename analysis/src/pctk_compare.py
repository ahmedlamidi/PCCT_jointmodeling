"""
PcTK 3.2 -- spectral response ablation + pileup cascade, compared against an ideal detector.

Reads the shipped PcTK tables directly; no MATLAB required.
First run extracts diag(m3_nCov3x3E) from the 4 GB .mat (~7 min) and caches it to ../cache/.

Validated: the E-domain binning below reproduces the shipped nCov3x3w to 3.3e-16.
"""
import os, numpy as np, h5py, scipy.io

from paths import PCTK, CACHE

# ==========================================================================
#  TOGGLES
# ==========================================================================
CFG = dict(
    # ---- acquisition -----------------------------------------------------
    N0            = 12_000.0,          # photons/pixel/view. NOT the 1.41e6 in v_N0.mat (unphysical)
    VIEWS_PER_SEC = 4_000.0,
    ETH           = [20., 50., 65., 80.],   # any count, any values -- rebinned in the E domain
    PATHS         = [('air', 0., 0.),
                     ('20 cm brain', 20., 0.),
                     ('20 cm brain + 2 cm bone', 20., 2.)],

    # ---- detector physics (no regeneration needed) ------------------------
    FLUOR         = 'full',   # 'full' | 'off' (charge sharing only) | 'escape_only'
    PIXELS        = '3x3',    # '3x3' (uniform illumination) | 'center' (drops cross-talk counts)

    # ---- pileup: NOT part of PcTK, cascaded on the deposited-energy pmf ----
    PILEUP        = True,
    TAU_NS        = 20.0,
    TRIGGER_KEV   = 5.0,      # depositions below the noise floor do not trigger
    N_EVENTS      = 4_000_000,
)
# ==========================================================================

NCOVE = os.path.join(PCTK, '5_refdata/dat_nCovE_ver3.2_dpix_225_dz_1600_r0_24_esig_2.0.mat')

def _build_cache():
    os.makedirs(CACHE, exist_ok=True)
    h = h5py.File(NCOVE, 'r')
    vEo = np.array(h['v_Eo']).ravel()
    if not os.path.exists(f'{CACHE}/diagE.npy'):
        print('extracting diag(nCovE), ~7 min ...')
        D = h['m3_nCov3x3E']
        np.save(f'{CACHE}/diagE.npy', np.stack([np.diag(np.array(D[i])) for i in range(D.shape[0])]))
    if not os.path.exists(f'{CACHE}/q_sym.npy'):
        def sym(a):
            b = a.reshape(len(a), 9, 191).copy()
            for p in (1, 5, 7): b[:, p] = b[:, 3]     # quadrant symmetry: only pixels 4,5,9 are valid
            for p in (0, 2, 6): b[:, p] = b[:, 8]
            b[:, :, vEo > 175] = 0.0                  # overflow accumulator, absent from the diagonal
            return b.reshape(len(a), 1719)
        np.save(f'{CACHE}/q_sym.npy',
                np.stack([sym(np.array(h[k])) for k in ('m2_SRE_q0', 'm2_SRE_q1', 'm2_SRE_q2')]))
    if not os.path.exists(f'{CACHE}/Pr3.npy'):
        np.save(f'{CACHE}/Pr3.npy', np.array(h['v_Pr_pq'])[:, :170])

def load():
    _build_cache()
    h = h5py.File(NCOVE, 'r')
    d = dict(diagE=np.load(f'{CACHE}/diagE.npy'), q=np.load(f'{CACHE}/q_sym.npy'),
             Pr=np.load(f'{CACHE}/Pr3.npy'),
             vEo=np.array(h['v_Eo']).ravel(), vE1=np.array(h['v_E1']).ravel())
    d['QE'] = d['Pr'].sum(0)                                   # interaction prob = 1 - transmission
    d['S']  = scipy.io.loadmat(f'{PCTK}/1_inputdata/pmf_S0_Al2.0_120kVp.mat')['m2_S0'][:170, 1]
    d['mu'] = np.loadtxt(f'{PCTK}/1_inputdata/m2_mukE.csv', delimiter=',', skiprows=1)[:170, 1:]
    return d

def responses(d, cfg):
    """R[E1, Eo]: counts registered per incident photon, by deposited energy."""
    Pr, q = d['Pr'], d['q']
    full = d['diagE']
    noK  = Pr.sum(0)[:, None] * q[0]                           # all interactions -> no-fluorescence branch
    esc  = Pr[0][:, None] * q[0] + (Pr[1] + Pr[2])[:, None] * q[1]
    R = {'full': full, 'off': noK, 'escape_only': esc}[cfg['FLUOR']].reshape(170, 9, 191)
    R = R[:, 4, :] if cfg['PIXELS'] == 'center' else R.sum(1)
    ideal = np.zeros((170, 191))
    for i, E in enumerate(d['vE1']):
        ideal[i, int(np.argmin(abs(d['vEo'] - E)))] = d['QE'][i]
    return R, ideal, full.reshape(170, 9, 191).sum(1)

def binner(d, cfg):
    eth = np.asarray(cfg['ETH'], float)
    idx = np.digitize(d['vEo'], eth) - 1                       # -1 => below lowest threshold, uncounted
    return lambda R: np.stack([R[:, idx == l].sum(1) for l in range(len(eth))], 1)

def pileup(dep, vEo, rate, tau, cfg, rng):
    m = vEo >= cfg['TRIGGER_KEV']
    p = dep[m] / dep[m].sum(); n = cfg['N_EVENTS']
    t = np.cumsum(rng.exponential(1.0 / rate, n))
    E = rng.choice(vEo[m], size=n, p=p)
    rec, i = [], 0
    while i < n:                                               # non-extending, energy-summing
        j = max(np.searchsorted(t, t[i] + tau, side='left'), i + 1)
        rec.append(E[i:j].sum()); i = j
    rec = np.array(rec); eth = np.asarray(cfg['ETH'], float)
    hi = np.append(eth[1:], np.inf)
    return np.array([((rec >= eth[l]) & (rec < hi[l])).sum() for l in range(len(eth))]) / t[-1]

def main(cfg=CFG):
    d = load(); rng = np.random.default_rng(0)
    R, R_ideal, R_ref = responses(d, cfg)
    tw = binner(d, cfg); N0, eth = cfg['N0'], np.asarray(cfg['ETH'], float)
    spec = lambda b, bo: d['S'] * np.exp(-d['mu'][:, 0] * b - d['mu'][:, 1] * bo)
    lbl  = ' '.join('[%g-%s)' % (eth[l], eth[l+1] if l < len(eth)-1 else 'inf') for l in range(len(eth)))

    print('N0=%g  FLUOR=%s  PIXELS=%s  windows %s keV' % (N0, cfg['FLUOR'], cfg['PIXELS'], lbl))
    for name, b, bo in cfg['PATHS']:
        s = spec(b, bo)
        wi, wm = N0 * (tw(R_ideal).T @ s), N0 * (tw(R).T @ s)
        print('\n--- %s (transmission %.4f) ---' % (name, s.sum()))
        print('  %-22s %s  tot %9.1f' % ('ideal', ' '.join('%9.1f' % x for x in wi), wi.sum()))
        print('  %-22s %s  tot %9.1f' % ('PcTK [%s/%s]' % (cfg['FLUOR'], cfg['PIXELS']),
                                         ' '.join('%9.1f' % x for x in wm), wm.sum()))
        print('  %-22s %s' % ('rel. bias',
              ' '.join('%8.1f%%' % x for x in 100 * (wm - wi) / np.maximum(wi, 1e-9))))
        if cfg['PILEUP']:
            dep  = R_ref.T @ s
            rate = N0 * cfg['VIEWS_PER_SEC'] * dep[d['vEo'] >= cfg['TRIGGER_KEV']].sum()
            wp   = pileup(dep, d['vEo'], rate, cfg['TAU_NS'] * 1e-9, cfg, rng) / cfg['VIEWS_PER_SEC']
            wn   = N0 * (tw(R_ref).T @ s)
            print('  %-22s %s  tot %9.1f   (%.2e cps/pix, R*tau=%.3f)'
                  % ('+pileup', ' '.join('%9.1f' % x for x in wp), wp.sum(), rate, rate * cfg['TAU_NS'] * 1e-9))
            print('  %-22s %s' % ('pileup change',
                  ' '.join('%8.1f%%' % x for x in 100 * (wp - wn) / np.maximum(wn, 1e-9))))

if __name__ == '__main__':
    main()

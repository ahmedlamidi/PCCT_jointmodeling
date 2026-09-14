"""Pulse pile-up: Monte Carlo reference + fast grid-interpolated operator.

PcTK covers the rate-INdependent half (spectral response). This is the
rate-dependent half, cascaded after it, per Cammin/Taguchi ordering:

    incident spectrum -> PcTK (charge sharing, escape) -> deposited-energy pmf
                      -> pile-up (this module) -> thresholds -> bin counts

Detector model: seminonparalyzable with nonzero pulse length.
  tau_pulse  energies arriving within this window sum into one recorded pulse
  tau_dead   total dead time after a trigger (>= tau_pulse)
  extending  True = paralyzable (late arrivals restart the dead time)

Returns MEAN AND COVARIANCE of the bin counts, not just the mean spectrum --
pile-up makes counts sub-Poisson and any coverage analysis needs the second moment.

The analytical model (Yang et al., Med Phys 2025, doi 10.1002/mp.17746) would
replace mc_frames() for bulk generation. It is paywalled here, so instead
build_grid() exploits the fact that the deposited spectrum is a function of the
two material path lengths only: run MC on a grid, interpolate. Same speed, no
formula needed. Drop the analytical model in later and validate against mc_frames().
"""
import numpy as np


def mc_frames(dep_pmf, vEo, rate, tau_pulse, tau_dead, eth, n_frames, frame_T,
              rng, extending=False, trigger_keV=5.0, max_events=4_000_000):
    """Simulate n_frames independent frames. Returns counts (n_frames, Nl)."""
    m = vEo >= trigger_keV                      # sub-noise-floor charge cannot trigger
    p = dep_pmf[m]
    if p.sum() <= 0:
        return np.zeros((n_frames, len(eth)))
    p = p / p.sum(); e = vEo[m]

    exp_ev = rate * frame_T * n_frames
    if exp_ev > max_events:                     # keep the run bounded; fewer frames
        n_frames = max(20, int(max_events / max(rate * frame_T, 1e-9)))
    eth = np.asarray(eth, float); hi = np.append(eth[1:], np.inf)
    out = np.zeros((n_frames, len(eth)))

    for fr in range(n_frames):
        n = rng.poisson(rate * frame_T)
        if n == 0:
            continue
        t = np.sort(rng.random(n)) * frame_T
        E = rng.choice(e, size=n, p=p)
        i = 0
        while i < n:
            if extending:                       # paralyzable: arrivals extend the veto
                j = i + 1
                end = t[i] + tau_dead
                while j < n and t[j] < end:
                    end = t[j] + tau_dead
                    j += 1
                k = np.searchsorted(t, t[i] + tau_pulse, 'left')
            else:                               # non-extending
                j = np.searchsorted(t, t[i] + tau_dead, 'left')
                k = np.searchsorted(t, t[i] + tau_pulse, 'left')
            k = max(k, i + 1); j = max(j, i + 1)
            rec = E[i:k].sum()                   # energies inside the pulse sum
            if rec >= eth[0]:
                out[fr, np.searchsorted(eth, rec, 'right') - 1] += 1
            i = j
    return out[:n_frames]


def stats(counts):
    """Mean, covariance, and Fano factor per bin."""
    mu = counts.mean(0)
    C  = np.cov(counts.T) if counts.shape[0] > 1 else np.zeros((len(mu), len(mu)))
    C  = np.atleast_2d(C)
    fano = np.divide(np.diag(C), mu, out=np.ones_like(mu), where=mu > 0)
    return mu, C, fano


def build_grid(d, cfg, R_full, taus_ns, brain, bone, vps, n_frames=600,
               seed=0, verbose=True):
    """MC over (brain, bone) for each tau. Returns dict of arrays shaped
    (n_tau, n_brain, n_bone, Nl) for the mean and (..., Nl, Nl) for covariance."""
    import time
    rng = np.random.default_rng(seed)
    vEo, S, mu, N0 = d['vEo'], d['S'], d['mu'], cfg['N0']
    eth = np.asarray(cfg['ETH'], float); Nl = len(eth)
    nt, nb, nc = len(taus_ns), len(brain), len(bone)
    MU = np.zeros((nt, nb, nc, Nl)); CV = np.zeros((nt, nb, nc, Nl, Nl))
    FA = np.zeros((nt, nb, nc, Nl)); RT = np.zeros((nb, nc))
    t0 = time.time()
    for ib, b in enumerate(brain):
        for ic, bo in enumerate(bone):
            sp  = S * np.exp(-mu[:, 0]*b - mu[:, 1]*bo)
            dep = R_full.T @ sp
            rate = N0 * vps * dep[vEo >= cfg['TRIGGER_KEV']].sum()
            RT[ib, ic] = rate
            for it, tn in enumerate(taus_ns):
                if tn <= 0:                              # no pile-up: analytic Poisson
                    idx = np.digitize(vEo, eth) - 1
                    m = N0 * np.array([dep[idx == l].sum() for l in range(Nl)])
                    MU[it, ib, ic] = m; CV[it, ib, ic] = np.diag(m); FA[it, ib, ic] = 1.0
                    continue
                c = mc_frames(dep, vEo, rate, tn*1e-9, tn*1e-9, eth,
                              n_frames, 1.0/vps, rng)
                m, C, f = stats(c)
                MU[it, ib, ic] = m; CV[it, ib, ic] = C; FA[it, ib, ic] = f
        if verbose:
            print('  brain %5.1f cm   %.0fs elapsed' % (b, time.time()-t0), flush=True)
    return dict(taus_ns=np.asarray(taus_ns, float), brain=np.asarray(brain, float),
                bone=np.asarray(bone, float), mean=MU, cov=CV, fano=FA, rate=RT)


class PileupOperator:
    """Bilinear interpolation of a build_grid() result. Vectorised over pixels."""
    def __init__(self, grid, tau_ns):
        self.g = grid
        self.it = int(np.argmin(np.abs(grid['taus_ns'] - tau_ns)))
        self.tau = grid['taus_ns'][self.it]

    def __call__(self, vb, vbo):
        """vb, vbo: arrays of path lengths (cm). -> mean (...,Nl), cov (...,Nl,Nl)."""
        gb, gc = self.g['brain'], self.g['bone']
        x = np.clip(np.interp(vb,  gb, np.arange(len(gb))), 0, len(gb)-1-1e-9)
        y = np.clip(np.interp(vbo, gc, np.arange(len(gc))), 0, len(gc)-1-1e-9)
        i0 = x.astype(int); j0 = y.astype(int); fx = (x-i0)[..., None]; fy = (y-j0)[..., None]
        M = self.g['mean'][self.it]
        m = ((M[i0, j0]*(1-fx) + M[i0+1, j0]*fx)*(1-fy) +
             (M[i0, j0+1]*(1-fx) + M[i0+1, j0+1]*fx)*fy)
        C = self.g['cov'][self.it]; fx2 = fx[..., None]; fy2 = fy[..., None]
        c = ((C[i0, j0]*(1-fx2) + C[i0+1, j0]*fx2)*(1-fy2) +
             (C[i0, j0+1]*(1-fx2) + C[i0+1, j0+1]*fx2)*fy2)
        return m, c

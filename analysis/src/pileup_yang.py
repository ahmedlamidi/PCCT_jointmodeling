"""Analytical pileup: Yang, Pelc & Wang, Med Phys 52(6):3658-3674 (2025).
doi 10.1002/mp.17746   Seminonparalyzable, nonzero pulse length.

Triangular pulse: rises to E at t_p, falls to 0 at T.
m-th order pileup = m extra pulses inside the dead time, P(m) = Poisson(lam*tau).
Output = peak of the summed signal during the dead time.
Counts from renewal theory; binned stats from a compound multinomial.
"""
import numpy as np


def tri(E, t, t_p, T):
    """Triangular pulse height at time t (t may be an array)."""
    up   = np.clip(t / t_p, 0, None) * (t < t_p)
    down = np.clip((T - t) / (T - t_p), 0, None) * (t >= t_p) * (t < T)
    return E * (up + down)


def snp_duration(E, Et, t_p, T):
    """Eq 18: time for the pulse to peak then decay back to the trigger threshold."""
    E = np.asarray(E, float)
    out = np.where(E > Et, t_p + (T - t_p) * (1.0 - np.clip(Et / np.maximum(E, 1e-12), 0, 1)), 0.0)
    return out


def pair_peak(Ea, Eb, dt, t_p, T):
    """Peak of two triangular pulses, second delayed by dt. Ea,Eb,dt broadcast."""
    a_at_b = tri(Ea, t_p + dt, t_p, T)      # pulse a evaluated at b's peak
    b_at_a = tri(Eb, t_p - dt, t_p, T)      # pulse b evaluated at a's peak
    return np.maximum(Ea + b_at_a, Eb + a_at_b)


class YangPileup:
    def __init__(self, vE, tau, t_p, T, Et, m_max=12, n_dt=24):
        self.vE, self.tau, self.t_p, self.T, self.Et = vE, tau, t_p, T, Et
        self.m_max, self.n_dt = m_max, n_dt
        self.dE = vE[1] - vE[0]

    def _pile_once(self, g, f, m, k):
        """Pile spectrum g (effective pulse) with one more incident pulse f.
        Gap between them ~ the k-th of m order statistics on [0, tau]."""
        dts = (np.arange(self.n_dt) + 0.5) / self.n_dt * self.tau
        a = k; b = m - k + 1                       # Beta(a,b) order statistic
        w = dts**(a-1) * (self.tau - dts)**(b-1)
        w = w / w.sum()
        E = self.vE
        out = np.zeros_like(g)
        nz_g = np.nonzero(g)[0]; nz_f = np.nonzero(f)[0]
        if len(nz_g) == 0 or len(nz_f) == 0:
            return g.copy()
        Ea = E[nz_g][:, None]; Eb = E[nz_f][None, :]
        P  = g[nz_g][:, None] * f[nz_f][None, :]
        for wi, dt in zip(w, dts):
            pk = pair_peak(Ea, Eb, dt, self.t_p, self.T)
            idx = np.clip(np.round(pk / self.dE).astype(int), 0, len(E) - 1)
            np.add.at(out, idx.ravel(), (wi * P).ravel())
        return out

    def spectrum(self, f0, lam):
        """Recorded pmf and P(m). f0 = incident deposited pmf on vE (sums to 1)."""
        lt = lam * self.tau
        Pm = np.exp(-lt) * lt**np.arange(self.m_max + 1) / \
             np.array([np.math.factorial(k) for k in range(self.m_max + 1)])
        Pm = Pm / Pm.sum()
        spec = np.zeros_like(f0); per_m = []
        for m in range(self.m_max + 1):
            g = f0.copy()
            for k in range(1, m + 1):
                g = self._pile_once(g, f0, m, k)
                s = g.sum()
                if s > 0: g /= s
            per_m.append(g)
            spec += Pm[m] * g
        s = spec.sum()
        return (spec / s if s > 0 else spec), Pm, per_m

    def p_zero_free(self, per_m, Pm, lam):
        """Eq 19-20: probability the next dead time is retriggered (free time = 0)."""
        P0 = 0.0
        for m in range(1, self.m_max + 1):
            g = per_m[m]
            snp = snp_duration(self.vE, self.Et, self.t_p, self.T)
            s = np.clip((self.tau - snp) / self.tau, 0, 1)
            P0 += Pm[m] * np.sum(g * (1.0 - s**m))
        return float(np.clip(P0, 0, 1))

    def count_stats(self, lam, P0, Tframe):
        """Eq 16-17 via renewal theory. Cycle = dead time + free time;
        free time is 0 w.p. P0, else Exp(lam)."""
        EF = (1 - P0) / lam
        VF = (1 - P0) * (1 + P0) / lam**2
        ER = self.tau + EF; VR = VF
        EN = Tframe / ER
        VN = Tframe * VR / ER**3
        return EN, VN

    def binned(self, spec, EN, VN, eth):
        """Eq 21-22: compound multinomial -> mean vector and covariance."""
        eth = np.asarray(eth, float); hi = np.append(eth[1:], np.inf)
        p = np.array([spec[(self.vE >= eth[l]) & (self.vE < hi[l])].sum()
                      for l in range(len(eth))])
        mu = EN * p
        C  = EN * (np.diag(p) - np.outer(p, p)) + VN * np.outer(p, p)
        return mu, C, p


def apply(dep_pmf, vE, rate, Tframe, eth, tau, t_p, T, Et, m_max=12):
    """One call: deposited-energy pmf (per incident photon) -> binned mean, cov."""
    tot = dep_pmf[vE >= Et].sum()
    if tot <= 0:
        return np.zeros(len(eth)), np.zeros((len(eth),) * 2), 0.0
    f0 = np.where(vE >= Et, dep_pmf, 0.0) / tot
    lam = rate
    Y = YangPileup(vE, tau, t_p, T, Et, m_max=m_max)
    spec, Pm, per_m = Y.spectrum(f0, lam)
    P0 = Y.p_zero_free(per_m, Pm, lam)
    EN, VN = Y.count_stats(lam, P0, Tframe)
    mu, C, p = Y.binned(spec, EN, VN, eth)
    return mu, C, P0

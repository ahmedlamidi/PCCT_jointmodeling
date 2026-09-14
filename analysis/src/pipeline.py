"""Forward pipeline: material path lengths -> binned PCD counts.

    incident spectrum -> PcTK (charge sharing, escape; rate-independent)
                      -> pile-up (Yang 2025; rate-dependent)
                      -> thresholds -> binned counts (mean + covariance)

Two branches share a noise seed, so paired samples differ only by detector physics:
    ideal      perfect thresholds, same QE, Poisson
    distorted  PcTK (+ optional pile-up), correlated covariance

Operator parameters exposed for p(S): FLUOR, PIXELS, ETH, tau, t_p, T, N0.
tau=0 disables pile-up.
"""
import numpy as np, h5py
from paths import PCTK
import pctk_compare as P
import pileup_yang as PY


class Forward:
    def __init__(self, cfg=None, tau_ns=0.0, t_p_ns=10.0, T_ns=30.0, vps=4000.0,
                 grid_brain=None, grid_bone=None, verbose=True, backend='mc',
                 mc_frames=250, pileup_mode='paralyzable'):
        # pileup_mode: 'paralyzable' (Bierme & Roessl 2012, Roessl et al 2016 --
        # what Morovati et al. 2025 use, so it is the default here) or
        # 'seminonparalyzable' (Yang et al. 2025). They agree below lam*tau ~ 0.1
        # and differ 3.4x at lam*tau = 2.45.
        # backend: 'mc' (Monte Carlo reference, trusted) or 'yang' (analytical,
        # fast but its Eq 16-20 were reconstructed from prose; error grows with
        # rate -- 2% at lam*tau=0.01, 6% at 0.1, 16% at 0.6, 37% at 2.4).
        # NOTE T_ns defaults to 30 >= pulse length 25, otherwise every pulse
        # retriggers itself on its own tail and bin 1 inflates ~38%.
        self.backend, self.mc_frames = backend, mc_frames
        self.pileup_mode = pileup_mode
        self.cfg = dict(P.CFG if cfg is None else cfg)
        self.d   = P.load()
        self.eth = np.asarray(self.cfg['ETH'], float); self.Nl = len(self.eth)
        R, Rideal, _ = P.responses(self.d, self.cfg)
        self.R, self.Rideal = R, Rideal
        tw = P.binner(self.d, self.cfg)
        self.Wf, self.Wi = tw(R), tw(Rideal)                 # (170, Nl)
        self.N0, self.vps = self.cfg['N0'], vps
        self.tau, self.t_p, self.T = tau_ns*1e-9, t_p_ns*1e-9, T_ns*1e-9

        # per-pixel (Nl x Nl) PcTK covariance, per incident energy.
        # PcTK's own thresholds come straight from the shipped table; any other
        # binning is rebinned from the full nCovE by rebin_cov (cached).
        PCTK_ETH = np.array([20., 50., 65., 80.])
        if self.Nl == 4 and np.allclose(self.eth, PCTK_ETH):
            hw = h5py.File(PCTK+'/1_inputdata/dat_nCovw_ver3.2_dpix_225_dz_1600_r0_24_esig_2.0_'
                           'Eth_20.0_50.0_65.0_80.0_keV.mat', 'r')
            Cfull = np.array(hw['m3_nCov3x3w'])              # (170, 36, 36)
        else:
            import rebin_cov
            Cfull = rebin_cov.load(self.eth)                 # (170, 9*Nl, 9*Nl)
        self.C36 = Cfull                                     # full neighbourhood covariance
        Cp = sum(Cfull[:, self.Nl*p:self.Nl*(p+1), self.Nl*p:self.Nl*(p+1)] for p in range(9))
        self.C4 = .5*(Cp + np.transpose(Cp, (0, 2, 1)))      # (170, Nl, Nl), name kept

        self.grid = None
        if self.tau > 0:
            gb = np.concatenate([np.arange(0, 8, 1.), np.arange(8, 40, 2.)]) \
                 if grid_brain is None else np.asarray(grid_brain, float)
            gc = np.linspace(0, 6, 13) if grid_bone is None else np.asarray(grid_bone, float)
            self._build_grid(gb, gc, verbose)

    # ---- spectra -----------------------------------------------------------
    def spec(self, vb, vbo):
        """(...,) path lengths -> (170, ...) attenuated spectrum."""
        vb = np.asarray(vb, float); vbo = np.asarray(vbo, float)
        return self.d['S'][:, None] * np.exp(-np.outer(self.d['mu'][:, 0], vb.ravel())
                                             - np.outer(self.d['mu'][:, 1], vbo.ravel()))

    # ---- no-pileup analytic path ------------------------------------------
    def _mean_cov_nopile(self, sp):
        mu_d = self.N0 * (self.Wf.T @ sp)                    # (Nl,B)
        mu_i = self.N0 * (self.Wi.T @ sp)
        C_d  = self.N0 * np.einsum('eb,eij->bij', sp, self.C4)
        return mu_d.T, C_d, mu_i.T

    # ---- pileup grid -------------------------------------------------------
    def _build_grid(self, gb, gc, verbose):
        import time
        rng = np.random.default_rng(0)
        MU = np.zeros((len(gb), len(gc), self.Nl))
        CV = np.zeros((len(gb), len(gc), self.Nl, self.Nl))
        t0 = time.time()
        for i, b in enumerate(gb):
            for j, bo in enumerate(gc):
                sp  = self.spec(b, bo)[:, 0]
                dep = self.R.T @ sp
                rate = self.N0 * self.vps * dep[self.d['vEo'] >= self.cfg['TRIGGER_KEV']].sum()
                if self.backend == 'mc':
                    import pileup_mc_ref as MR
                    c, _ = MR.mc_pulse_train(dep, self.d['vEo'], rate, self.tau,
                                             self.t_p, self.T, self.cfg['TRIGGER_KEV'],
                                             self.eth, self.mc_frames, 1.0/self.vps, rng,
                                             mode=self.pileup_mode)
                    mu = c.mean(0); C = np.cov(c.T) if len(c) > 1 else np.diag(mu)
                    C = np.atleast_2d(C)
                else:
                    mu, C, _ = PY.apply(dep, self.d['vEo'], rate, 1.0/self.vps, self.eth,
                                        self.tau, self.t_p, self.T, self.cfg['TRIGGER_KEV'])
                MU[i, j] = mu; CV[i, j] = C
            if verbose: print('  pileup grid brain %5.1f cm  %.0fs' % (b, time.time()-t0), flush=True)
        self.grid = dict(brain=gb, bone=gc, mean=MU, cov=CV)

    def _interp(self, vb, vbo):
        g = self.grid; gb, gc = g['brain'], g['bone']
        x = np.clip(np.interp(vb,  gb, np.arange(len(gb))), 0, len(gb)-1-1e-9)
        y = np.clip(np.interp(vbo, gc, np.arange(len(gc))), 0, len(gc)-1-1e-9)
        i0 = x.astype(int); j0 = y.astype(int)
        fx = (x-i0)[:, None]; fy = (y-j0)[:, None]
        M = g['mean']
        mu = ((M[i0,j0]*(1-fx)+M[i0+1,j0]*fx)*(1-fy) + (M[i0,j0+1]*(1-fx)+M[i0+1,j0+1]*fx)*fy)
        C = g['cov']; fx2 = fx[..., None]; fy2 = fy[..., None]
        cv = ((C[i0,j0]*(1-fx2)+C[i0+1,j0]*fx2)*(1-fy2) + (C[i0,j0+1]*(1-fx2)+C[i0+1,j0+1]*fx2)*fy2)
        return mu, cv

    # ---- public ------------------------------------------------------------
    def mean_cov(self, vb, vbo):
        """-> distorted mean (B,Nl), distorted cov (B,Nl,Nl), ideal mean (B,Nl)."""
        vb = np.asarray(vb, float).ravel(); vbo = np.asarray(vbo, float).ravel()
        sp = self.spec(vb, vbo)
        mu_d, C_d, mu_i = self._mean_cov_nopile(sp)
        if self.grid is not None:
            mu_d, C_d = self._interp(vb, vbo)
        return mu_d, C_d, mu_i

    def sample(self, vb, vbo, rng):
        """Paired realisation with a SHARED noise seed."""
        mu_d, C_d, mu_i = self.mean_cov(vb, vbo)
        B = mu_d.shape[0]
        z = rng.standard_normal((B, self.Nl))
        w, V = np.linalg.eigh(C_d)
        L = V @ (np.sqrt(np.clip(w, 0, None))[..., None] * np.transpose(V, (0, 2, 1)))
        y_d = mu_d + np.einsum('bij,bj->bi', L, z)
        y_i = mu_i + np.sqrt(np.maximum(mu_i, 0)) * z          # same z
        return np.maximum(y_d, 0), np.maximum(y_i, 0), mu_d, mu_i

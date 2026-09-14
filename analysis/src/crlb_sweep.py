"""Estimator efficiency vs operating point.

Question: is the plain ML material decomposition already at the Cramer-Rao bound?
If it is, no estimator can do better, so every gain a learned corrector reports
is prior rather than recovered information.

Outputs figures/crlb_sweep.png and outputs/crlb_sweep.npz.
"""
import numpy as np, h5py, scipy.io, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import PCTK, FIGS, OUT
import pctk_compare as P

rng = np.random.default_rng(0)
d   = P.load(); cfg = dict(P.CFG)
R_full, R_ideal, _ = P.responses(d, cfg)
tw  = P.binner(d, cfg)
Wf, Wi = tw(R_full), tw(R_ideal)                      # (170, Nl)
S, mu  = d['S'], d['mu']
N0  = cfg['N0']

hw = h5py.File(PCTK+'/1_inputdata/dat_nCovw_ver3.2_dpix_225_dz_1600_r0_24_esig_2.0_Eth_20.0_50.0_65.0_80.0_keV.mat','r')
Cw = np.array(hw['m3_nCov3x3w'])
C4 = sum(Cw[:, 4*p:4*p+4, 4*p:4*p+4] for p in range(9)); C4 = .5*(C4+np.transpose(C4,(0,2,1)))

def spec(V):                                          # V:(2,B) -> (170,B)
    return S[:,None]*np.exp(-mu[:,0:1]@V[0:1] - mu[:,1:2]@V[1:2])
def f(V, W):    return N0*(W.T @ spec(V))             # (Nl,B)
def J(V, W):                                          # (Nl,2,B)
    sp = spec(V)
    return np.stack([-N0*(W.T @ (mu[:,k:k+1]*sp)) for k in range(2)], 1)

def fisher(v, W, C):
    Jv = J(v[:,None], W)[:,:,0]
    return Jv.T @ np.linalg.solve(C, Jv)

def ml_batch(Y, v0, W, C, nit=80):
    """Vectorised Gauss-Newton, one fit per column of Y."""
    V = np.repeat(v0[:,None], Y.shape[1], 1).astype(float)
    Ci = np.linalg.inv(C)
    for _ in range(nit):
        r  = Y - f(V, W)                              # (Nl,B)
        Jv = J(V, W)                                  # (Nl,2,B)
        A  = np.einsum('ikb,ij,jlb->klb', Jv, Ci, Jv)
        g  = np.einsum('ikb,ij,jb->kb',  Jv, Ci, r)
        det = A[0,0]*A[1,1]-A[0,1]*A[1,0]
        det = np.where(np.abs(det)<1e-12, np.nan, det)
        st = np.stack([( A[1,1]*g[0]-A[0,1]*g[1])/det,
                       (-A[1,0]*g[0]+A[0,0]*g[1])/det])
        V = V + np.clip(st, -2, 2)
        V = np.clip(V, -5, 80)
    return V

BRAIN = [10., 20., 30.]; BONE = np.linspace(0., 5., 11); NR = 4000
res = {k: np.zeros((len(BRAIN), len(BONE))) for k in
       ('crlb_i','crlb_f','sd_ml','eff','penalty','bias')}
for a, b in enumerate(BRAIN):
    for c, bo in enumerate(BONE):
        v  = np.array([b, bo]); sp = spec(v[:,None])[:,0]
        Ci = np.diag(f(v[:,None], Wi)[:,0])                  # ideal: Poisson
        Cf = N0*np.einsum('e,eij->ij', sp, C4)               # PcTK: correlated
        ci = np.sqrt(np.linalg.inv(fisher(v, Wi, Ci))[1,1])
        cf = np.sqrt(np.linalg.inv(fisher(v, Wf, Cf))[1,1])
        y0 = f(v[:,None], Wf)[:,0]
        Y  = y0[:,None] + np.linalg.cholesky(Cf) @ rng.standard_normal((4, NR))
        V  = ml_batch(Y, v, Wf, Cf)
        ok = np.all(np.isfinite(V), 0) & (np.abs(V[1]-bo) < 40)
        sd = V[1, ok].std()
        res['crlb_i'][a,c]=ci; res['crlb_f'][a,c]=cf; res['sd_ml'][a,c]=sd
        res['eff'][a,c]=cf/sd; res['penalty'][a,c]=cf/ci
        res['bias'][a,c]=V[1,ok].mean()-bo
    print('brain %2.0f cm done'%b, flush=True)

np.savez(OUT+'/crlb_sweep.npz', BRAIN=BRAIN, BONE=BONE, **res)
fig, ax = plt.subplots(1,3, figsize=(15,4.2))
for a,b in enumerate(BRAIN):
    ax[0].plot(BONE, res['penalty'][a], 'o-', label='%g cm brain'%b)
    ax[1].plot(BONE, res['eff'][a],     'o-', label='%g cm brain'%b)
    ax[2].semilogy(BONE, res['crlb_f'][a], 'o-', label='PcTK, %g cm'%b)
    ax[2].semilogy(BONE, res['crlb_i'][a], 's--', color=ax[2].lines[-1].get_color(), alpha=.5)
ax[0].set_title('noise penalty  sd(PcTK)/sd(ideal)'); ax[0].axhline(1,color='k',lw=.7)
ax[1].set_title('estimator efficiency  CRLB / sd(ML)'); ax[1].axhline(1,color='k',lw=1.2,ls='--')
ax[1].set_ylim(0,1.4)
ax[2].set_title('CRLB on bone (solid PcTK, dashed ideal)')
for a_ in ax: a_.set_xlabel('bone thickness (cm)'); a_.legend(fontsize=8); a_.grid(alpha=.3)
ax[0].set_ylabel('ratio'); ax[2].set_ylabel('sd (cm)')
fig.suptitle('Is ML decomposition already optimal?  (dashed line at 1.0 = at the bound)')
fig.tight_layout(); fig.savefig(FIGS+'/stats_crlb_efficiency.png', dpi=130, bbox_inches='tight')

print('\n  bone   penalty(20cm brain)   efficiency   bias(cm)')
for c,bo in enumerate(BONE):
    print('  %4.1f      %.3f              %.3f       %+.4f'
          %(bo, res['penalty'][1,c], res['eff'][1,c], res['bias'][1,c]))

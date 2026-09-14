"""End-to-end demo: reference phantom -> sinogram -> distorted counts
-> material decomposition -> FBP reconstruction.

Two decompositions of the SAME distorted counts:
    uncorrected  fitted with the IDEAL detector model  (cross-talk ignored)
    corrected    fitted with the PcTK model            (cross-talk modelled)

Everything is chunked over views; peak memory stays near 1 GB.

    python3 demo_end2end.py [--nview 2000] [--chunk 100] [--tau 0]
"""
import argparse, numpy as np, scipy.io, time, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import PCTK, FIGS, OUT
from pipeline import Forward

ap = argparse.ArgumentParser()
ap.add_argument('--nview', type=int, default=2000)
ap.add_argument('--chunk', type=int, default=100)
ap.add_argument('--row',   type=int, default=3)
ap.add_argument('--tau',   type=float, default=0.0)
ap.add_argument('--nit',   type=int, default=40)
A = ap.parse_args()

NVIEW, NPIX, FOV, Nch = A.nview, 512, 250.0, 1854
R_, Rd_, dpix = 600.0, 1080.0, 0.225
gam  = (np.arange(Nch) - (Nch-1)/2) * (dpix/Rd_)
beta = np.arange(NVIEW) * 2*np.pi/NVIEW


def rebin_fbp(g):
    """Fan-to-parallel rebinning + ramp-filtered backprojection. g: (Nch, NVIEW)."""
    out = np.empty_like(g)
    for i in range(Nch):
        out[i] = np.interp((beta - gam[i]) % (2*np.pi), beta, g[i], period=2*np.pi)
    t = R_*np.sin(gam); tu = np.linspace(t[0], t[-1], Nch)
    p = np.stack([np.interp(tu, t, out[:, k]) for k in range(NVIEW)], 1)
    dt = tu[1]-tu[0]; Npad = 1 << int(np.ceil(np.log2(2*Nch)))
    filt = 2*np.abs(np.fft.fftfreq(Npad))/dt
    pf = np.real(np.fft.ifft(np.fft.fft(p, Npad, axis=0)*filt[:, None], axis=0))[:Nch]
    ax = (np.arange(NPIX)-(NPIX-1)/2)*(FOV/NPIX)
    X, Y = np.meshgrid(ax, ax); img = np.zeros((NPIX, NPIX))
    for k in range(NVIEW):
        img += np.interp(X*np.cos(beta[k])+Y*np.sin(beta[k]), tu, pf[:, k], left=0, right=0)
    return img*(np.pi/NVIEW)


def decompose(Y, W, Ci, F, nit):
    """Vectorised Gauss-Newton ML. Y:(Nl,B) -> V:(2,B) path lengths in cm."""
    N0, S, mu = F.N0, F.d['S'], F.d['mu']
    V = np.stack([np.full(Y.shape[1], 10.0), np.full(Y.shape[1], 0.5)])
    for _ in range(nit):
        sp = S[:, None]*np.exp(-np.outer(mu[:, 0], V[0]) - np.outer(mu[:, 1], V[1]))
        r  = Y - N0*(W.T @ sp)
        J  = np.stack([-N0*(W.T @ (mu[:, k:k+1]*sp)) for k in range(2)], 1)
        Aq = np.einsum('ikb,ij,jlb->klb', J, Ci, J)
        g  = np.einsum('ikb,ij,jb->kb',  J, Ci, r)
        det = Aq[0,0]*Aq[1,1] - Aq[0,1]*Aq[1,0]
        det = np.where(np.abs(det) < 1e-12, np.nan, det)
        st = np.stack([( Aq[1,1]*g[0]-Aq[0,1]*g[1])/det,
                       (-Aq[1,0]*g[0]+Aq[0,0]*g[1])/det])
        V = np.clip(V + np.clip(np.nan_to_num(st), -3, 3), 0, 60)
    return V


t0 = time.time()
F  = Forward(tau_ns=A.tau)
sv = scipy.io.loadmat(PCTK+'/1_inputdata/m4_sino_v.mat')['m4_sino_v']
vb  = sv[0, :, A.row, :NVIEW].astype(float)        # (Nch, NVIEW) true brain path, cm
vbo = sv[1, :, A.row, :NVIEW].astype(float)
rng = np.random.default_rng(0)

# fixed covariances, evaluated at a representative mid-object operating point
sp0 = F.spec(np.array([15.]), np.array([1.]))[:, 0]
Ci_ideal = np.linalg.inv(np.diag(F.N0*(F.Wi.T @ sp0)))
Ci_pctk  = np.linalg.inv(F.N0*np.einsum('e,eij->ij', sp0, F.C4))

Vun = np.zeros((2, Nch, NVIEW)); Vco = np.zeros((2, Nch, NVIEW))
for s in range(0, NVIEW, A.chunk):
    e = min(s+A.chunk, NVIEW)
    b, bo = vb[:, s:e].ravel(), vbo[:, s:e].ravel()
    yd, _, _, _ = F.sample(b, bo, rng)
    Y = np.ascontiguousarray(yd.T)                  # (Nl, B)
    Vun[:, :, s:e] = decompose(Y, F.Wi, Ci_ideal, F, A.nit).reshape(2, Nch, e-s)
    Vco[:, :, s:e] = decompose(Y, F.Wf, Ci_pctk,  F, A.nit).reshape(2, Nch, e-s)
    if s % (A.chunk*5) == 0:
        print('  views %4d/%d   %.0fs' % (e, NVIEW, time.time()-t0), flush=True)

print('\nsinogram-domain path-length error (cm):')
stats = {}
for nm, V in (('uncorrected', Vun), ('corrected', Vco)):
    eb, ebo = V[0]-vb, V[1]-vbo
    stats[nm] = (eb.mean(), np.sqrt((eb**2).mean()), ebo.mean(), np.sqrt((ebo**2).mean()))
    print('  %-12s brain bias %+7.3f rmse %6.3f  |  bone bias %+7.3f rmse %6.3f' % (nm, *stats[nm]))

rec = {}
for nm, (a, b) in dict(truth=(vb, vbo), uncorrected=(Vun[0], Vun[1]),
                       corrected=(Vco[0], Vco[1])).items():
    rec[nm+'_brain'] = rebin_fbp(np.ascontiguousarray(a))
    rec[nm+'_bone']  = rebin_fbp(np.ascontiguousarray(b))
    print('  recon %-12s %.0fs' % (nm, time.time()-t0), flush=True)
np.savez_compressed(OUT+'/demo_end2end.npz', **rec)

print('\nimage-domain error vs truth (density units):')
for nm in ('uncorrected', 'corrected'):
    for mat in ('brain', 'bone'):
        d = rec[nm+'_'+mat] - rec['truth_'+mat]
        print('  %-12s %-6s  bias %+8.4f   rmse %7.4f' % (nm, mat, d.mean(), np.sqrt((d**2).mean())))

fig, ax = plt.subplots(2, 3, figsize=(13, 8.6))
for c, nm in enumerate(['truth', 'uncorrected', 'corrected']):
    for r, mat in enumerate(['brain', 'bone']):
        vmax = np.percentile(rec['truth_'+mat], 99.5)
        ax[r, c].imshow(rec[nm+'_'+mat], cmap='gray', vmin=0, vmax=vmax)
        ax[r, c].set_title('%s — %s' % (nm, mat), fontsize=10); ax[r, c].axis('off')
fig.suptitle('Same distorted counts, decomposed with the wrong vs the right detector model')
fig.tight_layout(); fig.savefig(FIGS+'/recon_decomposition_wrong_vs_right.png', dpi=130, bbox_inches='tight')
print('\nwrote figures/demo_end2end.png  outputs/demo_end2end.npz  (%.0fs)' % (time.time()-t0))

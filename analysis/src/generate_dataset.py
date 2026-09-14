"""Paired training data: distorted PCD counts (input) + ideal counts (label).

  python3 generate_dataset.py --nviews 200 --tau 0    --out ds_tau00
  python3 generate_dataset.py --nviews 200 --tau 20   --out ds_tau20

Writes <out>.npz:
    X    (nview, Nrow, Nch, Nl) float32  distorted counts   -- network input
    Y    (nview, Nrow, Nch, Nl) float32  ideal counts       -- label
    MU_D (nview, Nrow, Nch, Nl) float32  distorted mean     -- noise-free target
    V    (nview, Nrow, Nch, 2)  float32  true path lengths (cm) -- quantitative GT
    meta operator parameters S

Both branches use the SAME standard-normal draw, so a pair differs only by
detector physics. Phantom line integrals come from PcTK's shipped m4_sino_v.
"""
import argparse, numpy as np, scipy.io, sys, time
from paths import PCTK, OUT
from pipeline import Forward
import pctk_compare as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nviews', type=int, default=200)
    ap.add_argument('--tau',    type=float, default=0.0, help='dead time, ns (0 = no pile-up)')
    ap.add_argument('--t_p',    type=float, default=10.0)
    ap.add_argument('--T',      type=float, default=25.0)
    ap.add_argument('--fluor',  default='full', choices=['full', 'off', 'escape_only'])
    ap.add_argument('--n0',     type=float, default=None)
    ap.add_argument('--seed',   type=int, default=0)
    ap.add_argument('--out',    default='dataset')
    a = ap.parse_args()

    cfg = dict(P.CFG); cfg['FLUOR'] = a.fluor
    if a.n0: cfg['N0'] = a.n0
    F = Forward(cfg=cfg, tau_ns=a.tau, t_p_ns=a.t_p, T_ns=a.T)
    rng = np.random.default_rng(a.seed)

    sv = scipy.io.loadmat(PCTK + '/1_inputdata/m4_sino_v.mat')['m4_sino_v']   # (2,Nch,Nrow,Nview)
    Nch, Nrow = sv.shape[1], sv.shape[2]
    nv = min(a.nviews, sv.shape[3]); Nl = F.Nl
    sh = (nv, Nrow, Nch, Nl)
    X = np.zeros(sh, np.float32); Y = np.zeros(sh, np.float32); MD = np.zeros(sh, np.float32)
    V = np.zeros((nv, Nrow, Nch, 2), np.float32)

    t0 = time.time()
    for iv in range(nv):
        vb  = sv[0, :, :, iv].T.ravel()          # (Nrow*Nch,)
        vbo = sv[1, :, :, iv].T.ravel()
        yd, yi, md, _ = F.sample(vb, vbo, rng)
        X[iv]  = yd.reshape(Nrow, Nch, Nl)
        Y[iv]  = yi.reshape(Nrow, Nch, Nl)
        MD[iv] = md.reshape(Nrow, Nch, Nl)
        V[iv, :, :, 0] = vb.reshape(Nrow, Nch)
        V[iv, :, :, 1] = vbo.reshape(Nrow, Nch)
        if iv % 25 == 0:
            print('  view %d/%d  %.0fs' % (iv, nv, time.time()-t0), flush=True)

    meta = dict(tau_ns=a.tau, t_p_ns=a.t_p, T_ns=a.T, fluor=a.fluor,
                N0=cfg['N0'], eth=np.asarray(cfg['ETH']), pixels=cfg['PIXELS'],
                seed=a.seed, phantom='PcTK m4_sino_v (brain/bone head)')
    p = OUT + '/' + a.out + '.npz'
    np.savez_compressed(p, X=X, Y=Y, MU_D=MD, V=V, **{'meta_'+k: v for k, v in meta.items()})
    print('\nwrote %s  X%s  %.0f MB  in %.0fs' %
          (p, sh, __import__('os').path.getsize(p)/1e6, time.time()-t0))
    print('  distorted mean over set %s' % np.round(X.reshape(-1, Nl).mean(0), 1))
    print('  ideal     mean over set %s' % np.round(Y.reshape(-1, Nl).mean(0), 1))


if __name__ == '__main__':
    main()

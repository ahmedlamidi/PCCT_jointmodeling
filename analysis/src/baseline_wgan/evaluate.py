"""Evaluate the WGAN baseline with the metrics Morovati et al. report: PSNR, SSIM, RMSE.

    python3 evaluate.py --ckpt ../../outputs/wgan_baseline3d_Y/ckpt.pt \
                        --arms baseline3d baseline3d_pu

Also reports the manifold residual, so the point estimate can be compared against
the diffusion posterior on the same physical-consistency measure. It cannot be
given a coverage number: the generator is deterministic, one output per input,
which is exactly the gap this project is filling.
"""
import argparse, json, os
import numpy as np
import torch

import sys
sys.path.insert(0, '..'); sys.path.insert(0, '../diffusion')
from paths import OUT
from data import DiffusionPatches
from manifold import Manifold
from models import Generator
from metrics import psnr, ssim



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arms', nargs='+', default=['baseline3d', 'baseline3d_pu'])
    ap.add_argument('--target', default='Y')
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--npatch', type=int, default=256)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    d = torch.load(a.ckpt, map_location=a.device)
    G = Generator(ch=d['ch'], base=d['cfg']['base']).to(a.device)
    G.load_state_dict(d['G']); G.eval()
    trained_on = d['cfg']['arm']
    print('checkpoint: arm=%s it=%d' % (trained_on, d['it']), flush=True)

    res, mf = {}, None
    for arm in a.arms:
        ds = DiffusionPatches(arm, 'test', a.patch, a.target)
        x, y = ds.fixed_eval_set(a.npatch)      # identical positions to the diffusion eval
        with torch.no_grad():
            p = G(torch.from_numpy(x).to(a.device)).cpu().numpy()
        P = ds.denorm_t(p.transpose(0, 2, 3, 1)).reshape(-1, ds.tgt_ch)
        Y = ds.denorm_t(y.transpose(0, 2, 3, 1)).reshape(-1, ds.tgt_ch)
        if mf is None:
            mf = Manifold.fit(Y)
        res[arm] = dict(rmse=float(np.sqrt(((P - Y) ** 2).mean())),
                        rmse_per_bin=np.sqrt(((P - Y) ** 2).mean(0)).tolist(),
                        psnr=float(psnr(P, Y)), ssim=float(ssim(P, Y)),
                        resid_pred=float(mf.residual(P).mean()),
                        resid_truth=float(mf.residual(Y).mean()))
        tag = 'MATCHED' if arm == trained_on else 'MISMATCH'
        r = res[arm]
        print('\n--- %s (%s) ---' % (arm, tag))
        print('  RMSE %.4f   PSNR %.2f dB   SSIM %.4f' % (r['rmse'], r['psnr'], r['ssim']))
        print('  manifold residual: prediction %.4f  vs truth %.4f'
              % (r['resid_pred'], r['resid_truth']))
        print('  coverage: N/A -- deterministic generator, no posterior')

    fp = os.path.join(OUT, 'wgan_eval_%s.json' % trained_on)
    with open(fp, 'w') as f:
        json.dump(dict(trained_on=trained_on, npatch=a.npatch, eval_seed=1234,
                       results=res), f, indent=2)
    print('\nwrote', fp)


if __name__ == '__main__':
    main()

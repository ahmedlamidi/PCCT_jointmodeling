"""Coverage under forward-model mismatch, for a model predicting BIN COUNTS.

Train on one operator setting, sample the posterior on both arms, and ask whether
the credible intervals still contain the truth at the stated rate.

    python3 coverage.py --ckpt ../../outputs/edm_baseline3d_Y/ckpt.pt \
                        --arms baseline3d baseline3d_pu --nsamp 64

Three things are measured, and the third is the one that matters:

 1  PER-BIN coverage      each energy bin separately. Standard, and WEAK -- a
                          model can hit 90% in all nine bins and still be wrong.
 2  JOINT coverage        the 9-vector as a whole, via Mahalanobis rank inside
                          the sample cloud on the 2-D manifold. No distributional
                          assumption; uses the empirical sample quantile.
 3  MANIFOLD residual     how far samples sit OFF the 2-D surface that physical
                          count vectors must lie on. Per-bin coverage is blind to
                          this by construction; see manifold.py.

Calibration on the MATCHED arm is a prerequisite, not a result. If the matched
arm is already miscalibrated the mismatch number means nothing.

Read the MATCHED-vs-MISMATCH difference, not the absolute number: at nsamp=64 a
perfectly calibrated model reads ~0.866 at the 0.90 level purely from the
empirical-quantile bias (see manifold.joint_coverage). The bias cancels between
arms at equal nsamp; it does not cancel in an absolute claim.
"""
import argparse, json, os
import numpy as np
import torch

import sys; sys.path.insert(0, '..')
from paths import OUT, FIGS
from data import DiffusionPatches
from unet import UNet
from edm import EDMPrecond, edm_sample
from manifold import Manifold, joint_coverage
# the SAME metric code as the WGAN baseline, so the two columns are comparable
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baseline_wgan'))
from metrics import psnr, ssim

LEVELS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


def load_net(ckpt, device):
    d = torch.load(ckpt, map_location=device)
    cfg = d['cfg']
    model = UNet(d['tgt_ch'] + d['cond_ch'], d['tgt_ch'], base=cfg['base'])
    net = EDMPrecond(model, sigma_data=cfg['sigma_data']).to(device)
    net.load_state_dict(d['ema'])
    net.eval()
    return net, d


@torch.no_grad()
def posterior(net, cond, tgt_ch, nsamp, steps, device, seed=0):
    B, _, H, W = cond.shape
    out = []
    for s in range(nsamp):
        g = torch.Generator(device=device).manual_seed(seed + s)
        out.append(edm_sample(net, (B, tgt_ch, H, W), cond=cond, steps=steps,
                              device=device, generator=g).cpu().numpy())
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arms', nargs='+', default=['baseline3d', 'baseline3d_pu'])
    ap.add_argument('--target', default='Y')
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--npatch', type=int, default=256)
    ap.add_argument('--nsamp', type=int, default=64)
    ap.add_argument('--steps', type=int, default=18)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    net, meta = load_net(a.ckpt, a.device)
    trained_on = meta['cfg']['arm']
    print('checkpoint: arm=%s target=%s it=%d' % (trained_on, meta['cfg']['target'], meta['it']),
          flush=True)

    res = {}
    mf = None
    for arm in a.arms:
        ds = DiffusionPatches(arm, 'test', a.patch, a.target)
        x, y = ds.fixed_eval_set(a.npatch)          # same seed -> same positions in every arm
        cond = torch.from_numpy(x).to(a.device)
        S = posterior(net, cond, ds.tgt_ch, a.nsamp, a.steps, a.device)
        Sp = ds.denorm_t(S.transpose(0, 1, 3, 4, 2)).reshape(a.nsamp, -1, ds.tgt_ch)
        Yp = ds.denorm_t(y.transpose(0, 2, 3, 1)).reshape(-1, ds.tgt_ch)

        if mf is None:                              # fit ONCE, on the matched arm's truth
            mf = Manifold.fit(Yp)
            print('manifold: 2 components explain %.4f%% of the truth variance'
                  % (100 * mf.explained()[1]))

        per_bin = np.zeros((len(LEVELS), ds.tgt_ch))
        for li, L in enumerate(LEVELS):
            lo = np.quantile(Sp, (1 - L) / 2, axis=0)
            hi = np.quantile(Sp, 1 - (1 - L) / 2, axis=0)
            per_bin[li] = ((Yp >= lo) & (Yp <= hi)).mean(0)

        jc = joint_coverage(mf.coords(Sp), mf.coords(Yp), LEVELS)
        r_s = mf.residual(Sp).mean()
        r_y = mf.residual(Yp).mean()

        pm = Sp.mean(0)                             # posterior mean ~ the MMSE estimate
        res[arm] = dict(per_bin=per_bin.tolist(), joint=jc.tolist(),
                        resid_samples=float(r_s), resid_truth=float(r_y),
                        resid_postmean=float(mf.residual(pm).mean()),
                        rmse=np.sqrt(((pm - Yp) ** 2).mean(0)).tolist(),
                        rmse_all=float(np.sqrt(((pm - Yp) ** 2).mean())),
                        psnr=float(psnr(pm, Yp)), ssim=float(ssim(pm, Yp)))
        tag = 'MATCHED' if arm == trained_on else 'MISMATCH'
        print('\n--- %s (%s) ---' % (arm, tag))
        print('  nominal      ' + '  '.join('%5.2f' % L for L in LEVELS))
        print('  per-bin mean ' + '  '.join('%5.3f' % v for v in per_bin.mean(1)))
        print('  JOINT        ' + '  '.join('%5.3f' % v for v in jc))
        print('  manifold residual: samples %.4f  vs truth %.4f  (ratio %.2f)'
              % (r_s, r_y, r_s / max(r_y, 1e-9)))
        print('  posterior mean: RMSE %.4f   PSNR %.2f dB   SSIM %.4f'
              % (res[arm]['rmse_all'], res[arm]['psnr'], res[arm]['ssim']))

    fp = os.path.join(OUT, 'coverage_%s_%s.json' % (trained_on, a.target))
    with open(fp, 'w') as f:
        json.dump(dict(levels=LEVELS.tolist(), trained_on=trained_on,
                       target=a.target, npatch=a.npatch, nsamp=a.nsamp, eval_seed=1234,
                       results=res), f, indent=2)
    print('\nwrote', fp)

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
        for k, (key, ttl) in enumerate([('per_bin', 'per-bin (marginal)'),
                                        ('joint', 'joint, on the 2-D manifold')]):
            ax[k].plot([0, 1], [0, 1], 'k--', lw=1, label='perfect calibration')
            for arm in a.arms:
                v = np.array(res[arm][key])
                v = v.mean(1) if v.ndim == 2 else v
                lab = arm + (' (matched)' if arm == trained_on else ' (MISMATCH)')
                ax[k].plot(LEVELS, v, 'o-', lw=1.8, label=lab)
            ax[k].set_xlabel('nominal coverage'); ax[k].set_ylabel('empirical coverage')
            ax[k].set_title(ttl); ax[k].grid(alpha=.3); ax[k].legend(fontsize=8)
        fig.suptitle('Coverage (trained on %s, target %s)' % (trained_on, a.target))
        fig.tight_layout()
        p = os.path.join(FIGS, 'stats_coverage_%s_%s.png' % (trained_on, a.target))
        fig.savefig(p, dpi=130, bbox_inches='tight')
        print('wrote', p)
    except Exception as e:
        print('plot skipped:', e)


if __name__ == '__main__':
    main()

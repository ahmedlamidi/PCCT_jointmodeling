"""WGAN baseline vs diffusion on one arm -- one table.

    python3 compare_baseline.py --arm baseline3d_pu_matched

Reads the two JSONs the SLURM jobs write:
    outputs/wgan_eval_<arm>.json         (baseline_wgan/evaluate.py)
    outputs/coverage_<arm>_Y.json        (diffusion/coverage.py)

Both are scored on the SAME held-out test patches (DiffusionPatches.fixed_eval_set,
same n and seed, drawn across every test phantom) with the SAME metric code
(baseline_wgan/metrics.py), so the columns are directly comparable. The script
refuses to print a table if the patch counts differ.

Both methods' outputs pass through the same DiffusionPatches.denorm_t, which clips
counts to [0, air count per bin] -- no ray can record more than the unattenuated
beam. Without that, a few tail samples dominated the diffusion posterior mean
(expm1 of a +5 sigma log-space value is ~1e6 counts).

How to read it: the WGAN is a point estimate trained with a lambda=1000 MSE anchor,
so it is EXPECTED to win on RMSE/PSNR -- that is what its loss optimises. The
diffusion model's posterior mean approximates the MMSE estimator. The rows only
the diffusion model can fill -- coverage -- are the contribution.
"""
import argparse, json, os, sys
from paths import OUT


def load(p):
    if not os.path.exists(p):
        sys.exit('missing %s -- has that job finished?' % p)
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='baseline3d_pu_matched')
    a = ap.parse_args()

    W = load(os.path.join(OUT, 'wgan_eval_%s.json' % a.arm))
    D = load(os.path.join(OUT, 'coverage_%s_Y.json' % a.arm))
    if W.get('npatch') != D.get('npatch') or W.get('eval_seed') != D.get('eval_seed'):
        sys.exit('evaluation sets differ (WGAN npatch=%s seed=%s, diffusion npatch=%s seed=%s)'
                 ' -- rerun one of them so both use the same patches'
                 % (W.get('npatch'), W.get('eval_seed'), D.get('npatch'), D.get('eval_seed')))
    w, d = W['results'][a.arm], D['results'][a.arm]
    L = D['levels']; i90 = L.index(0.9)
    per_bin90 = sum(d['per_bin'][i90]) / len(d['per_bin'][i90])

    rows = [
        ('RMSE (counts)',                 '%.4f' % w['rmse'],        '%.4f' % d['rmse_all']),
        ('PSNR (dB)',                     '%.2f' % w['psnr'],        '%.2f' % d['psnr']),
        ('SSIM',                          '%.4f' % w['ssim'],        '%.4f' % d['ssim']),
        ('manifold residual, estimate',   '%.4f' % w['resid_pred'],  '%.4f' % d['resid_postmean']),
        ('manifold residual, truth',      '%.4f' % w['resid_truth'], '%.4f' % d['resid_truth']),
        ('manifold residual, samples',    'n/a',                     '%.4f' % d['resid_samples']),
        ('90% coverage, per bin (mean)', 'n/a (deterministic)',     '%.3f' % per_bin90),
        ('90% coverage, joint',          'n/a (deterministic)',     '%.3f' % d['joint'][i90]),
    ]
    print('\n%s  --  %d test patches, same for both\n' % (a.arm, W['npatch']))
    print('%-32s %-22s %-22s' % ('', 'WGAN (point estimate)', 'diffusion (post. mean)'))
    print('-' * 76)
    for r in rows:
        print('%-32s %-22s %-22s' % r)
    print('\nCoverage at nsamp=%d reads a few points low even for a perfectly calibrated'
          ' model\n(0.886 at nominal 0.90 with 256 samples) -- see diffusion/manifold.py.'
          % D['nsamp'])

    out = os.path.join(OUT, 'compare_%s.json' % a.arm)
    with open(out, 'w') as f:
        json.dump(dict(arm=a.arm, npatch=W['npatch'], wgan=w, diffusion=d), f, indent=2)
    print('wrote', out)


if __name__ == '__main__':
    main()

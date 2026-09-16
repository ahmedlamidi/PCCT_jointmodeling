"""Does the predicted uncertainty track the actual error? Per energy bin, from a
slice file written by ssim_eval.py. No GPU, no torch.

    python3 plot_unc_vs_err.py <slice_phXXX_rowYY.npz> --out <dir>

Per bin, over the pixels of one region (head by default: label < half the air
count; --region air / all for the others):

  * Spearman and Pearson correlation between the posterior std and |error|.
    Spearman is the one to read: the relation should be monotonic, not linear.
  * A reliability curve: pixels sorted into 10 groups by predicted std; for each
    group the mean std (x) against the RMSE of the actual error (y). A calibrated
    model lies on the diagonal: where it says +/-1 count, it is off by 1 count.
    Above the diagonal = overconfident, below = too cautious.
  * For scale, the Poisson noise on the INPUT, sqrt(label): the noise the model
    has to see through. It is NOT a floor on the posterior std -- the model
    predicts the noise-free label and pools 256 pixels x 9 bins per patch, so its
    std sits far below sqrt(label). (An earlier version of this docstring called
    it a floor; that was wrong.)

Writes <dir>/reliability_bin<N>.png per bin, <dir>/reliability_all_bins.png, and
prints a table. Numbers are for ONE slice of ONE phantom; the std is a 16-sample
estimate, so the scatter within a group is partly estimation noise.
"""
import argparse, os, re
import numpy as np
from scipy.stats import spearmanr, pearsonr


def bin_label(b, eth=(20, 30, 40, 50, 60, 70, 80, 90, 100)):
    hi = eth[b + 1] if b + 1 < len(eth) else eth[b] + 10
    return '%g-%g keV' % (eth[b], hi - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('slice')
    ap.add_argument('--out', required=True)
    ap.add_argument('--region', default='head', choices=['head', 'air', 'all'])
    ap.add_argument('--groups', type=int, default=10)
    a = ap.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sl = np.load(a.slice)
    nsamp, row = int(sl['nsamp']), int(sl['row'])
    m = re.search(r'slice_ph(\d+)_', os.path.basename(a.slice))
    ph = int(m.group(1)) if m else -1
    if float(sl['std'].max()) == 0:
        raise SystemExit('std is 0 everywhere: a deterministic model has no uncertainty to check')
    os.makedirs(a.out, exist_ok=True)
    C = sl['label'].shape[-1]
    rows, curves = [], []
    print('%-12s %7s %7s %8s %8s   %s' % ('bin', 'spear', 'pears', 'RMSE', 'meanstd', 'RMSE/std per std-decile (low -> high)'))
    for b in range(C):
        L, M, S = (np.asarray(sl[k][:, :, b], np.float64) for k in ('label', 'mean', 'std'))
        head = L < 0.5 * L.max()
        reg = {'head': head, 'air': ~head, 'all': np.ones(L.shape, bool)}[a.region]
        e, s, l = np.abs(M - L)[reg], S[reg], L[reg]
        sp = spearmanr(s, e).correlation; pe = pearsonr(s, e)[0]
        order = np.argsort(s); groups = np.array_split(order, a.groups)
        xs = np.array([s[g].mean() for g in groups])
        ys = np.array([np.sqrt((e[g] ** 2).mean()) for g in groups])
        ps = np.array([np.sqrt(l[g]).mean() for g in groups])      # Poisson noise on the input
        rows.append((b, sp, pe, np.sqrt((e ** 2).mean()), s.mean()))
        curves.append((xs, ys))
        print('%-12s %7.3f %7.3f %8.3f %8.3f   %s' % (bin_label(b), sp, pe, rows[-1][3], rows[-1][4],
              ' '.join('%.2f' % (y / x) for x, y in zip(xs, ys))))

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        sub = np.random.default_rng(0).choice(len(s), min(20000, len(s)), replace=False)
        ax[0].scatter(s[sub], e[sub], s=2, alpha=0.15, color='C0')
        ax[0].set_xlabel('predicted std (counts)'); ax[0].set_ylabel('|error| (counts)')
        ax[0].set_title('per pixel (20k shown)   Spearman %.3f' % sp, fontsize=10)
        ax[0].grid(alpha=0.3)
        top = max(xs.max(), ys.max(), ps.max()) * 1.1
        ax[1].plot([0, top], [0, top], 'k--', lw=1, label='calibrated (RMSE = std)')
        ax[1].plot(xs, ys, 'o-', color='C0', label='diffusion, %d samples' % nsamp)
        ax[1].plot(xs, ps, 's:', color='0.5', ms=4, label='input Poisson noise sqrt(label), for scale')
        ax[1].set_xlabel('mean predicted std in group (counts)'); ax[1].set_ylabel('RMSE of actual error (counts)')
        ax[1].set_title('reliability: %d groups by predicted std' % a.groups, fontsize=10)
        ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
        fig.suptitle('%s -- test phantom %d, row %d, %s pixels' % (bin_label(b), ph, row, a.region))
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, 'reliability_bin%d.png' % (b + 1)), dpi=130, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    top = max(max(x.max(), y.max()) for x, y in curves) * 1.05
    ax.plot([0, top], [0, top], 'k--', lw=1, label='calibrated')
    for (b, *_), (xs, ys) in zip(rows, curves):
        ax.plot(xs, ys, 'o-', ms=3, lw=1.2, label=bin_label(b), color=plt.cm.viridis(b / (C - 1)))
    ax.set_xlabel('mean predicted std in group (counts)'); ax.set_ylabel('RMSE of actual error (counts)')
    ax.set_title('reliability, all bins -- %s pixels, phantom %d row %d' % (a.region, ph, row), fontsize=10)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'reliability_all_bins.png'), dpi=130, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()

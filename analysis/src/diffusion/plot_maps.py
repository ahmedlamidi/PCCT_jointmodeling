"""Uncertainty map and error map of one sinogram slice, one figure each, per bin.
Reads a slice file written by ssim_eval.py; no GPU, no torch.

    python3 plot_maps.py <slice_phXXX_rowYY.npz> --out <dir>
    python3 plot_maps.py <ssim_wgan_.../slice_...npz> --out <dir> --model_label WGAN
    python3 plot_maps.py <slice.npz> --out <dir> --head_only

--head_only: colour limits and the printed numbers come from the HEAD pixels only,
i.e. where the clean label is below half the bin's air count (the maximum of the
label in that bin). Without it the air at the sinogram edges (~2000 counts against
~20 in the head) sets the scale and the head is invisible. Air pixels are then
greyed out.

Per bin, in <dir>/bin<N>_<lo>-<hi>keV/:
    uncertainty_map.png   posterior std, counts  (skipped when std is 0 everywhere,
                          i.e. a deterministic model such as the WGAN)
    error_map.png         output - label, counts; error and std share one colour scale
    zscore_map.png        |output - label| / std  (skipped without std). For a
                          calibrated posterior about 95% of pixels are below 2.

The whole sinogram: all views x all channels. Colour limits are the 99th percentile
of |error| and of std over the sinogram, whichever is larger, so the two maps can
be read against each other. Std is the blended value over overlapping patches
(see ssim_eval.py): display only.
"""
import argparse, os, re
import numpy as np


def bin_label(b, eth=(20, 30, 40, 50, 60, 70, 80, 90, 100)):
    hi = eth[b + 1] if b + 1 < len(eth) else eth[b] + 10
    return '%g-%g keV' % (eth[b], hi - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('slice')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bins', type=int, nargs='+', default=[0, 4, 8])
    ap.add_argument('--model_label', default='diffusion')
    ap.add_argument('--head_only', action='store_true')
    a = ap.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sl = np.load(a.slice)
    nsamp, row = int(sl['nsamp']), int(sl['row'])
    m = re.search(r'slice_ph(\d+)_', os.path.basename(a.slice))
    ph = int(m.group(1)) if m else -1
    nview, nch = sl['label'].shape[:2]
    has_std = float(sl['std'].max()) > 0
    written = []

    def image(path, A, title, cmap, vmin, vmax, cbar, mask=None):
        fig, ax = plt.subplots(figsize=(8, 5.2))
        if mask is not None:
            A = np.ma.masked_where(~mask, A)
            cmap = plt.get_cmap(cmap).copy(); cmap.set_bad('0.6')
        im = ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                       interpolation='nearest', extent=[0, nch, nview, 0])
        fig.colorbar(im, ax=ax, label=cbar)
        ax.set_xlabel('detector channel'); ax.set_ylabel('view (acquisition order)')
        ax.set_title(title, fontsize=10)
        fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)
        written.append(path)

    for b in a.bins:
        lab = bin_label(b)
        d = os.path.join(a.out, 'bin%d_%s' % (b + 1, lab.replace(' ', '')))
        os.makedirs(d, exist_ok=True)
        L, M, S = (np.asarray(sl[k][:, :, b], np.float64) for k in ('label', 'mean', 'std'))
        err = M - L
        head = L < 0.5 * L.max() if a.head_only else np.ones(L.shape, bool)
        e_, s_ = err[head], S[head]
        rmse = float(np.sqrt((e_ ** 2).mean()))
        vmax = max(np.percentile(np.abs(e_), 99), np.percentile(s_, 99) if has_std else 0, 1e-6)
        where = '%s, test phantom %d, detector row %d%s' % (
            lab, ph, row, '  [head only: label < half air; air greyed]' if a.head_only else '')
        mk = head if a.head_only else None
        who = '%s (%d sample%s)' % (a.model_label, nsamp, '' if nsamp == 1 else 's')

        image(os.path.join(d, 'error_map.png'), err,
              'error map: %s output - clean label   (RMSE %.3f counts)\n%s' % (who, rmse, where),
              'RdBu_r', -vmax, vmax, 'counts', mk)
        if has_std:
            image(os.path.join(d, 'uncertainty_map.png'), S,
                  'uncertainty map: posterior std of %s   (mean std %.3f counts)\n%s'
                  % (who, s_.mean(), where), 'magma', 0, vmax, 'counts', mk)
            z = np.abs(err) / np.maximum(S, 1e-9)
            frac = float((z[head] <= 2).mean())
            image(os.path.join(d, 'zscore_map.png'), z,
                  '|error| / std   (%.1f%% of pixels below 2; calibrated would be ~95%%)\n%s'
                  % (100 * frac, where), 'viridis', 0, 4, '|error| / std', mk)
            print('%-12s RMSE %.3f  mean std %.3f  |err|/std<=2: %.1f%%  median z %.2f  (%d%% of pixels)'
                  % (lab, rmse, s_.mean(), 100 * frac, np.median(z[head]), round(100 * head.mean())))
        else:
            print('%-12s RMSE %.3f  (no std: deterministic model)' % (lab, rmse))
    for p in written:
        print('wrote', p)


if __name__ == '__main__':
    main()

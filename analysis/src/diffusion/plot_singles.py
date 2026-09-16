"""Every plot of one sinogram slice as its OWN figure -- no multi-panel overviews.
Reads a slice file written by ssim_eval.py (full run or preview); no GPU, no torch.

    python3 plot_singles.py <slice_phXXX_rowYY.npz> --out <dir>             # full run
    python3 plot_singles.py <..._preview/slice_ph000_row16.npz> --out <dir> --preview

Per bin (default bins 1, 5, 9) it writes <dir>/bin<N>_<lo>-<hi>keV/:

    01_sinogram_input.png       whole sinogram, -log(counts / air), no overlays
    02_sinogram_label.png
    03_sinogram_output.png
    04_crop_input.png           channels c0..c1 (default centre +/- 200), counts
    05_crop_label.png
    06_crop_output.png
    07_crop_error.png           output - label, counts
    08_crop_std.png             posterior std (not written with --preview: 2 samples)
    09_timeseries.png           one channel over all views: input, label, output (+/- 2 std)
    10_timeseries_residual.png  minus the label, with RMSE per curve
    11_profile.png              one view across all channels (log axis)
    12_profile_residual.png

The same numbers as ssim_eval.figure and plot_profiles.plot, drawn one per file:
the same colour scales (crops: the label's 1-99th percentile; error and std share
one scale), the same default channel (centre + 300) and view (nview // 4).
--preview: the std from 2 samples means nothing, so no std figure and no band.
"""
import argparse, os, re
import numpy as np


def bin_label(b, eth, closed=True):
    hi = eth[b + 1] if b + 1 < len(eth) else (eth[b] + eth[b] - eth[b - 1] if closed else None)
    return '%g-%g keV' % (eth[b], hi - 1) if hi else '%g+ keV' % eth[b]


def _rmse(z):
    return float(np.sqrt((z ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('slice')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bins', type=int, nargs='+', default=[0, 4, 8], help='bin indices 0-8')
    ap.add_argument('--eth', type=float, nargs='+', default=[20, 30, 40, 50, 60, 70, 80, 90, 100],
                    help='bin thresholds, for the labels (default: the 9 closed bins)')
    ap.add_argument('--channel', type=int, default=None, help='time-series channel (default centre + 300)')
    ap.add_argument('--view', type=int, default=None, help='profile view (default nview // 4)')
    ap.add_argument('--crop', type=int, nargs=2, default=None, help='channel range of the crops')
    ap.add_argument('--preview', action='store_true', help='2-sample preview: no std figure, no band')
    ap.add_argument('--model_label', default='diffusion posterior mean')
    a = ap.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sl = np.load(a.slice)
    nsamp, row = int(sl['nsamp']), int(sl['row'])
    m = re.search(r'slice_ph(\d+)_', os.path.basename(a.slice))
    ph = int(m.group(1)) if m else -1
    nview, nch = sl['label'].shape[:2]
    ch = nch // 2 + 300 if a.channel is None else a.channel
    view = nview // 4 if a.view is None else a.view
    c0, c1 = a.crop if a.crop else (nch // 2 - 200, nch // 2 + 200)
    out_lab = '%s (%d sample%s%s)' % (a.model_label, nsamp, '' if nsamp == 1 else 's',
                                      ', PREVIEW' if a.preview else '')
    where = 'test phantom %d, detector row %d' % (ph, row)
    written = []

    def save(fig, path):
        fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)
        written.append(path)

    def image(path, A, title, cmap, vmin, vmax, cbar, extent, marks=False):
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        im = ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                       interpolation='nearest', extent=extent)
        fig.colorbar(im, ax=ax, label=cbar)
        ax.set_xlabel('detector channel'); ax.set_ylabel('view (acquisition order)')
        ax.set_title(title, fontsize=10)
        save(fig, path)

    def lines(path, x, curves, title, xlab, ylab, band=None, logy=False, symlog=False):
        fig, ax = plt.subplots(figsize=(10, 4.2))
        if band is not None:
            lo_b, hi_b = band
            if logy:    # clip at the smallest plotted value, NOT a fixed count: bin 1
                        # through the head is ~0.001-0.01 counts, and a 0.5 clip drew a
                        # false band from ~0.01 up to 0.5 there
                pos = np.concatenate([y[y > 0] for y, *_ in curves])
                lo_b = np.maximum(lo_b, pos.min() if pos.size else 1e-6)
            ax.fill_between(x, lo_b, hi_b, color='C0', alpha=0.25, lw=0, label='output $\\pm$2 std')
        for y, colour, lw, lab in curves:
            ax.plot(x, y, color=colour, lw=lw, label=lab)
        if logy: ax.set_yscale('log')
        if symlog: ax.set_yscale('symlog', linthresh=10)
        ax.axhline(0, color='k', lw=0.6) if (symlog or ylab.startswith('minus')) else None
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='best')
        save(fig, path)

    for b in a.bins:
        lab = bin_label(b, a.eth)
        d = os.path.join(a.out, 'bin%d_%s' % (b + 1, lab.replace(' ', '')))
        os.makedirs(d, exist_ok=True)
        I, L, M, S = (np.asarray(sl[k][:, :, b], np.float64) for k in ('input', 'label', 'mean', 'std'))
        if a.preview:
            S = np.zeros_like(S)
        head = '%s, %s' % (lab, where)

        # 01-03: whole sinograms as line integrals (in counts the air swamps the head)
        air = L.max()
        li = lambda z: -np.log(np.maximum(z, 0.5) / air)
        lo, hi = np.percentile(li(L), [0.5, 99.5])
        full = [0, nch, nview, 0]
        for n, (A, t) in enumerate([(I, 'distorted input'), (L, 'clean label'), (M, out_lab)], 1):
            image(os.path.join(d, '%02d_sinogram_%s.png' % (n, ['input', 'label', 'output'][n - 1])),
                  li(A), '%s -- whole sinogram\n%s' % (t, head), 'gray', lo, hi,
                  '-log(counts / air)', full, marks=True)

        # 04-08: the centre crop in counts, error and std on one shared scale
        cr = lambda z: z[:, c0:c1]
        ext = [c0, c1, nview, 0]
        lo, hi = np.percentile(cr(L), [1, 99])
        err = cr(M) - cr(L)
        mm = max(np.percentile(np.abs(err), 99), np.percentile(cr(S), 99), 1e-6)
        for n, (A, t) in enumerate([(cr(I), 'distorted input'), (cr(L), 'clean label'), (cr(M), out_lab)], 4):
            image(os.path.join(d, '%02d_crop_%s.png' % (n, ['input', 'label', 'output'][n - 4])),
                  A, '%s -- channels %d-%d\n%s' % (t, c0, c1 - 1, head), 'gray', lo, hi, 'counts', ext)
        image(os.path.join(d, '07_crop_error.png'), err,
              'error: output - label (RMSE %.3f counts) -- channels %d-%d\n%s' % (_rmse(err), c0, c1 - 1, head),
              'RdBu_r', -mm, mm, 'counts', ext)
        if not a.preview:
            image(os.path.join(d, '08_crop_std.png'), cr(S),
                  'posterior std -- channels %d-%d\n%s' % (c0, c1 - 1, head), 'magma', 0, mm, 'counts', ext)

        # 09-12: one channel over all views, one view across all channels
        band = lambda m_, s_: None if a.preview else (m_ - 2 * s_, m_ + 2 * s_)
        v = np.arange(nview); c_ = np.arange(nch)
        i, l, mo, s = I[:, ch], L[:, ch], M[:, ch], S[:, ch]
        lines(os.path.join(d, '09_timeseries.png'), v,
              [(i, '0.55', 1.0, 'distorted input'), (l, 'k', 1.5, 'clean label'), (mo, 'C0', 1.2, out_lab)],
              'time series: channel %d over all %d views\n%s' % (ch, nview, head),
              'view (acquisition order)', 'counts', band(mo, s))
        lines(os.path.join(d, '10_timeseries_residual.png'), v,
              [(i - l, '0.55', 0.9, 'input - label  (RMSE %.3f)' % _rmse(i - l)),
               (mo - l, 'C0', 0.9, 'output - label  (RMSE %.3f)' % _rmse(mo - l))],
              'time series residual: channel %d\n%s' % (ch, head), 'view (acquisition order)',
              'minus label (counts)', None if a.preview else (-2 * s, 2 * s))
        i, l, mo, s = I[view], L[view], M[view], S[view]
        lines(os.path.join(d, '11_profile.png'), c_,
              [(i, '0.55', 1.0, 'distorted input'), (l, 'k', 1.5, 'clean label'), (mo, 'C0', 1.2, out_lab)],
              'profile: view %d across all %d channels (log axis)\n%s' % (view, nch, head),
              'detector channel', 'counts', band(mo, s), logy=True)
        lines(os.path.join(d, '12_profile_residual.png'), c_,
              [(i - l, '0.55', 0.9, 'input - label  (RMSE %.3f)' % _rmse(i - l)),
               (mo - l, 'C0', 0.9, 'output - label  (RMSE %.3f)' % _rmse(mo - l))],
              'profile residual: view %d (symmetric log axis)\n%s' % (view, head), 'detector channel',
              'minus label (counts)', None if a.preview else (-2 * s, 2 * s), symlog=True)

    for p in written:
        print('wrote', p)
    print('%d figures' % len(written))


if __name__ == '__main__':
    main()

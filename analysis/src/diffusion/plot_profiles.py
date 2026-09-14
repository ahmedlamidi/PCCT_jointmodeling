"""One sinogram, and input / clean label / model output traced through it: along one
detector channel over all views (the view axis is acquisition order, so this is a
time series), and along one view across all channels.

Reads a slice file written by ssim_eval.py -- no GPU, no torch:

    python3 plot_profiles.py ../../outputs/ssim_<train>_Y_on_<arm>/slice_ph000_row16.npz
    python3 plot_profiles.py <slice.npz> --bin 4 --channel 1200 --view 45

ssim_eval.py also calls plot() for its figure phantom, one figure per --vis_bins.

The shaded band is the model output (posterior mean) +/- 2 posterior std. That std
is blended across overlapping patches -- display only. If the model is calibrated
the clean label should sit inside the band most of the time; the count printed on
the residual panel is a visual check, not a coverage result (coverage.py is).

Default channel is 300 right of centre: the centre channel passes through the
middle of the head at every view, so its series is nearly flat; an off-centre
channel sweeps through different tissue as the gantry turns. Default view is
nview // 4 (view 0 would draw its marker on the image border).

The sinogram images are shown as -log(counts / air), air = the label's maximum in
that bin: in counts the air (~2000) swamps the head (~10). The profile panel uses a
log axis for the same reason.
"""
import argparse, os
import numpy as np


def _rmse(z):
    return float(np.sqrt((z ** 2).mean()))


def plot(sl, b, channel=None, view=None, label=None, path='profiles.png', title='',
         other=None, other_label='WGAN', main_label=None):
    """other: a second slice dict (e.g. the WGAN's) whose 'mean' is drawn as an extra
    sinogram panel and an extra line in both traces. main_label names sl's output."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    M, S, L, I = (np.asarray(sl[k][:, :, b], np.float64) for k in ('mean', 'std', 'label', 'input'))
    nview, nch = L.shape
    ch = nch // 2 + 300 if channel is None else int(channel)
    view = nview // 4 if view is None else int(view)
    label = label or 'bin index %d' % b

    O = None if other is None else np.asarray(other['mean'][:, :, b], np.float64)
    short = main_label or 'model output'
    tops = [(I, 'distorted input'), (L, 'clean label'), (M, main_label or 'model output (posterior mean)')]
    if O is not None:
        tops.append((O, other_label))
    fig = plt.figure(figsize=(15 + 4 * (O is not None), 14))
    gs = fig.add_gridspec(5, len(tops), height_ratios=[2.3, 1.6, 0.85, 1.6, 0.85], hspace=0.5, wspace=0.35)
    # Images as line integrals: in counts, air (~2000) swamps the head (~10) and the
    # object is black. Air = the label's maximum in this bin (the crop includes air).
    air = L.max()
    li = lambda z: -np.log(np.maximum(z, 0.5) / air)
    lo, hi = np.percentile(li(L), [0.5, 99.5])
    for c, (img, t) in enumerate(tops):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(li(img), cmap='gray', vmin=lo, vmax=hi, aspect='auto', interpolation='nearest')
        ax.axvline(ch, color='C1', lw=1.2); ax.axhline(view, color='C2', lw=1.2)
        ax.set_title(t, fontsize=10); ax.set_xlabel('detector channel')
        if c == 0: ax.set_ylabel('view')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='-log(counts / air)')

    def trace(row, x, i, l, m, s, o, xlab, what, colour, logy=False):
        ax = fig.add_subplot(gs[row, :]); axr = fig.add_subplot(gs[row + 1, :], sharex=ax)
        if logy:        # air and head differ ~200x in counts: log axis shows both
            ax.set_yscale('log'); axr.set_yscale('symlog', linthresh=10)
        if s.max() > 0:     # no std (WGAN, or a preview): no band and no legend entry for one
            ax.fill_between(x, np.maximum(m - 2 * s, 0.5 if logy else -np.inf), m + 2 * s,
                            color='C0', alpha=0.25, lw=0, label='output $\\pm$2 std')
        ax.plot(x, i, color='0.55', lw=1.0, label='distorted input')
        ax.plot(x, l, color='k', lw=1.5, label='clean label')
        ax.plot(x, m, color='C0', lw=1.2, label=short)
        if o is not None:
            ax.plot(x, o, color='C3', lw=1.2, label=other_label)
        ax.set_ylabel('counts'); ax.legend(ncol=5, fontsize=8, loc='upper right')
        ax.set_title(what, fontsize=10, color=colour, loc='left')
        if s.max() > 0:
            axr.fill_between(x, -2 * s, 2 * s, color='C0', alpha=0.2, lw=0)
        axr.axhline(0, color='k', lw=0.8)
        axr.plot(x, i - l, color='0.55', lw=0.9, label='input - label   RMSE %.2f' % _rmse(i - l))
        axr.plot(x, m - l, color='C0', lw=0.9, label='%s - label  RMSE %.2f' % (short, _rmse(m - l)))
        if o is not None:
            axr.plot(x, o - l, color='C3', lw=0.9, label='%s - label  RMSE %.2f' % (other_label, _rmse(o - l)))
        if s.max() > 0:     # a deterministic model (WGAN) has no std: no band, no count
            inside = int((np.abs(m - l) <= 2 * s).sum())
            axr.text(0.005, 0.95, 'label inside $\\pm$2 std: %d of %d points' % (inside, len(x)),
                     transform=axr.transAxes, va='top', fontsize=8, bbox=dict(fc='white', alpha=0.7, lw=0))
        axr.set_ylabel('minus label'); axr.set_xlabel(xlab); axr.legend(ncol=2, fontsize=8, loc='upper right')

    v = np.arange(nview)
    trace(1, v, I[:, ch], L[:, ch], M[:, ch], S[:, ch], None if O is None else O[:, ch],
          'view (acquisition order)',
          'time series: detector channel %d over all %d views (orange line above)' % (ch, nview), 'C1')
    c_ = np.arange(nch)
    trace(3, c_, I[view], L[view], M[view], S[view], None if O is None else O[view],
          'detector channel',
          'profile: view %d across all %d channels (green line above; log axis)' % (view, nch), 'C2',
          logy=True)
    fig.suptitle('%s   %s, detector row %d' % (title, label, int(sl['row'])), fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)
    print('  wrote', path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('slice', help='slice_phXXX_rowYY.npz written by ssim_eval.py')
    ap.add_argument('--bin', type=int, default=0, help='bin index 0-8 (0 = lowest energy)')
    ap.add_argument('--channel', type=int, default=None, help='default: centre + 300')
    ap.add_argument('--view', type=int, default=None, help='default: nview // 4')
    ap.add_argument('--out', default=None)
    ap.add_argument('--other', default=None,
                    help="a second slice file to overlay, e.g. the WGAN's "
                         "(outputs/ssim_wgan_<train>_Y_on_<arm>/slice_...npz)")
    ap.add_argument('--other_label', default='WGAN')
    a = ap.parse_args()
    sl = np.load(a.slice)
    out = a.out or os.path.splitext(a.slice)[0] + '_profiles_bin%d.png' % a.bin
    plot(sl, a.bin, a.channel, a.view, None, out, os.path.basename(a.slice),
         other=np.load(a.other) if a.other else None, other_label=a.other_label)


if __name__ == '__main__':
    main()

"""Windowed SSIM for the diffusion posterior mean -- on detector patches and on
reconstructed slices -- and a picture of where the error sits in the sinogram.

    python3 ssim_eval.py                        # both domains + figure (needs a GPU)
    python3 ssim_eval.py --parts patch          # patch domain only (quick)
    python3 ssim_eval.py --arm baseline3d_matched   # the matched model on another arm

WHY. The SSIM in coverage.py (metrics.ssim) is ONE global window over all 256
patches and all nine bins at once. The spread of the counts across bins and rays
(hundreds) dwarfs the error (RMSE ~4 counts), so it reads ~0.9999 for any sensible
prediction. This uses local-window SSIM (metrics.ssim_windowed: Wang et al. 2004,
doi 10.1109/TIP.2003.819861, 11x11 Gaussian window), one energy bin at a time.

A  PATCH DOMAIN. The same 256 test patches coverage.py scores (fixed_eval_set,
   seed 1234), the same seeds (sample s draws with seed s over the whole batch, as
   coverage.posterior does) and the same nsamp. So the posterior mean is the one
   coverage.py scored, up to GPU non-determinism; its RMSE is printed next to the
   coverage JSON's as a check. Each bin's 16x16 patch is one image; the data range
   L per bin is the unattenuated air count, the physical ceiling of a count.

B  IMAGE DOMAIN. One axial slice per test phantom. A slice is one detector row,
   and FBP needs that row in every view, so the model runs on the 16-row band of
   the detector plane around that row: 231 patches per view at stride 8 along the
   channels, 41,580 per phantom (the whole plane would be 124,200). Patches are
   blended along the channels with stitch.py's Hann weights. Each bin's row
   becomes line integrals -log(max(counts, 0.5) / air) and is reconstructed by
   FBP (parallel or fan, read from the manifest) with a Hann-apodised ramp cut at
   the image Nyquist (projector.fbp_parallel). The slice is scored against the
   FBP of the clean label, with L = the label image's max - min per bin. Both go
   through the same FBP, so a common scale error (the known ~1.85 mu scaling)
   cancels in SSIM exactly.

   Why the apodised filter: with the plain ramp, a NOISY count sinogram
   reconstructs as noise. Measured on this repo's old noisy-label arm
   (baseline3d_pu, test phantom 0, row 16): the FBP of the noise-free distorted
   mean MU_D correlates 0.93-0.95 with a soft/bone fit, the FBP of the noisy label
   0.03-0.15. The noisy INPUT and a 16-sample posterior mean are noisy sinograms,
   so the plain ramp would score mostly amplified noise.

   Why floor 0.5 counts (my choice, not from a paper): a zero count gives
   -log(1e-3/air) ~ 14 against ~3 for a typical ray, and FBP spreads that spike
   over the image. The fraction of sinogram values at the floor is printed and
   saved per phantom; if it is not ~0 for the label and the mean, distrust B.

   The posterior mean here averages --nsamp_recon samples (default 16), not 256:
   B has ~500x the patches of A. A 16-sample mean still carries Monte-Carlo noise
   of (posterior variance)/16, which LOWERS its SSIM relative to the exact
   posterior mean; a deterministic point estimate (the WGAN) has none. Raise
   --nsamp_recon if time allows. The JSON records the value.

   Both domains also score the distorted INPUT against the label ("input" column):
   the SSIM of doing no correction at all, so the model's number has a reference.

C  FIGURE. For one phantom, a crop of the slice's sinogram in a few bins:
   distorted input, clean label, posterior mean, error (mean - label) and
   posterior std, in counts. Error and std share one colour scale. The std is
   E[x^2] - E[x]^2 after blending overlapping patches -- display only.

Each phantom's slice is saved as it finishes (outputs/ssim_<train>_Y_on_<arm>/),
so a resubmit after a TIMEOUT continues where it stopped. --resample redraws.

STATUS: see the "Checked" block at the end of this docstring.

Checked (on the development machine, no torch, 2026-09-14):
  - metrics.ssim_windowed against skimage.metrics.structural_similarity.
  - projector.Geometry.fbp unchanged (bit for bit, window=None) by the
    fbp_parallel split; fbp_parallel reconstructs a forward_parallel phantom
    (corr 0.995 plain ramp, 0.993 Hann).
  - Hann window on the old noisy arm, row 16, soft/bone-fit correlation of the
    FBP, bins 1/5/9: noisy label 0.06-0.90 -> 0.78-0.96, noisy input
    0.09-0.12 -> 0.74-0.84, noise-free MU_D 0.93-0.95 -> 0.85-0.91.
  - the band/stitch/line-integral/FBP path end to end, with a stand-in "model"
    that returns its input: the result must equal the same computation done
    directly on the full arrays.
NOT executed: the torch sampling path (Sampler). The first HiPerGator run is its
first run.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, '..', 'baseline_wgan'))
from paths import OUT, FIGS
from data import DiffusionPatches
from stitch import hann2d, _starts
from metrics import ssim_windowed
from projector import Geometry

LI_FLOOR = 0.5           # counts; floor before -log (see docstring, B)
FBP_WINDOW = 'hann'      # ramp x Hann to the image Nyquist (projector.fbp_parallel)


class Sampler:
    """The trained EDM model -- the only torch-dependent part of this file, so the
    rest can be tested with a stand-in: any callable (cond, seed) -> sample."""

    def __init__(self, ckpt, device=None, steps=18):
        import torch
        from coverage import load_net
        from edm import edm_sample
        self.torch, self._sample, self.steps = torch, edm_sample, steps
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net, meta = load_net(ckpt, self.device)
        self.tgt_ch, self.trained_on, self.it = meta['tgt_ch'], meta['cfg']['arm'], meta['it']

    def __call__(self, cond, seed):
        """cond (B, 9, p, p) normalised -> one normalised sample (B, tgt_ch, p, p).
        One generator per call, seeded once: the same as coverage.posterior."""
        t = self.torch
        c = t.from_numpy(np.ascontiguousarray(cond, np.float32)).to(self.device)
        g = t.Generator(device=self.device).manual_seed(int(seed))
        x = self._sample(self.net, (c.shape[0], self.tgt_ch) + tuple(c.shape[2:]), cond=c,
                         steps=self.steps, device=self.device, generator=g)
        return x.cpu().numpy()


def bin_labels(man):
    e, closed = man['eth'], man.get('closed_top_bin', False)
    out = []
    for b in range(len(e)):
        hi = e[b + 1] if b + 1 < len(e) else (e[b] + e[b] - e[b - 1] if closed else None)
        out.append('%g-%g keV' % (e[b], hi - 1) if hi else '%g+ keV' % e[b])
    return out


def input_counts(ds, x):
    """Normalised condition (..., C) -> distorted counts, inverting DiffusionPatches._norm_x."""
    return np.expm1(x * ds.xs + ds.xm)


# ---------------------------------------------------------------- A: patches
def patch_posterior(ds, sampler, npatch, nsamp):
    x, y = ds.fixed_eval_set(npatch)
    acc = 0.0
    for s in range(nsamp):
        acc = acc + ds.denorm_t(sampler(x, s).transpose(0, 2, 3, 1)).astype(np.float64)
    return (acc / nsamp, ds.denorm_t(y.transpose(0, 2, 3, 1)),
            input_counts(ds, x.transpose(0, 2, 3, 1)))          # each (N, p, p, C) counts


def run_patch(a, ds, get_sampler, wd, labels, trained_on):
    f = os.path.join(wd, 'patch_postmean.npz')
    if os.path.exists(f) and not a.resample:
        z = np.load(f)
        if int(z['nsamp']) != a.nsamp or int(z['npatch']) != a.npatch:
            sys.exit('%s has nsamp=%d npatch=%d; pass --resample or match them'
                     % (f, int(z['nsamp']), int(z['npatch'])))
        pm, yt, xin = z['pm'], z['y'], z['x']
        print('patch domain: reusing', f)
    else:
        print('patch domain: %d patches x %d samples' % (a.npatch, a.nsamp), flush=True)
        pm, yt, xin = patch_posterior(ds, get_sampler(), a.npatch, a.nsamp)
        np.savez(f, pm=pm.astype(np.float32), y=yt, x=xin.astype(np.float32),
                 nsamp=a.nsamp, npatch=a.npatch)
    C = yt.shape[-1]
    L = ds.tmax if ds.tmax is not None else yt.reshape(-1, C).max(0) - yt.reshape(-1, C).min(0)
    chw = lambda z: z.transpose(0, 3, 1, 2)
    s_m = ssim_windowed(chw(pm), chw(yt), L)                    # (N, C)
    s_x = ssim_windowed(chw(xin), chw(yt), L)
    rmse = float(np.sqrt(((pm - yt) ** 2).mean()))

    print('\n--- PATCH domain: %d test patches, per-bin 16x16 windowed SSIM, L = air count ---'
          % len(pm))
    print('  %-12s %9s %9s %9s' % ('bin', 'mean', 'p10', 'input'))
    for b in range(C):
        print('  %-12s %9.4f %9.4f %9.4f' % (labels[b], s_m[:, b].mean(),
                                            np.percentile(s_m[:, b], 10), s_x[:, b].mean()))
    print('  %-12s %9.4f %9s %9.4f' % ('all bins', s_m.mean(), '', s_x.mean()))
    cf = os.path.join(OUT, 'coverage_%s_Y.json' % trained_on)
    chk = ''
    if os.path.exists(cf):
        cj = json.load(open(cf))
        r = cj['results'].get(a.arm or trained_on)
        if r and cj.get('nsamp') == a.nsamp and cj.get('npatch') == a.npatch:
            chk = '   (coverage.py: %.4f -- should agree to a few decimals)' % r['rmse_all']
    print('  posterior-mean RMSE %.4f counts%s' % (rmse, chk))
    return dict(npatch=int(len(pm)), nsamp=a.nsamp, data_range=np.asarray(L).tolist(),
                ssim_per_bin=s_m.mean(0).tolist(), ssim_p10_per_bin=np.percentile(s_m, 10, 0).tolist(),
                ssim=float(s_m.mean()), ssim_input_per_bin=s_x.mean(0).tolist(),
                ssim_input=float(s_x.mean()), rmse=rmse)


# ---------------------------------------------------------------- B: slices
def slice_posterior(ds, fi, sampler, row, stride, nsamp, batch):
    """Posterior mean and std of detector row `row`, every view, from the patches of
    the 16-row band around it. Returns counts, each (nview, nch, C)."""
    d = np.load(ds.files[fi])
    X, Y = d['X'], d['Y']
    nview, nrow, nch, _ = X.shape
    p, C = ds.p, ds.tgt_ch
    i0 = int(np.clip(row - p // 2, 0, nrow - p)); rr = row - i0
    Xb = np.ascontiguousarray(X[:, i0:i0 + p])
    inp, lab = X[:, row].copy(), Y[:, row].copy()
    del X, Y, d
    w = hann2d(p)[rr]                                  # blend weights along the channels
    pos = [(v, j) for v in range(nview) for j in _starts(nch, p, stride)]
    S1 = np.zeros((nview, nch, C)); S2 = np.zeros_like(S1); W = np.zeros((nview, nch))
    nchunk = (len(pos) + batch - 1) // batch
    print('  phantom %d: rows %d-%d, row %d reconstructed; %d patches x %d samples'
          % (fi, i0, i0 + p - 1, row, len(pos), nsamp), flush=True)
    t0 = time.time()
    for ci in range(nchunk):
        chunk = pos[ci * batch:(ci + 1) * batch]
        cond = np.stack([ds._norm_x(Xb[v, :, j:j + p]).transpose(2, 0, 1)
                         for v, j in chunk]).astype(np.float32)
        m1 = np.zeros((len(chunk), p, C)); m2 = np.zeros_like(m1)
        for s in range(nsamp):
            seed = (fi * 100_000 + ci) * 1000 + s
            smp = ds.denorm_t(sampler(cond, seed).transpose(0, 2, 3, 1))[:, rr].astype(np.float64)
            m1 += smp; m2 += smp ** 2
        for n, (v, j) in enumerate(chunk):
            S1[v, j:j + p] += w[:, None] * m1[n] / nsamp
            S2[v, j:j + p] += w[:, None] * m2[n] / nsamp
            W[v, j:j + p] += w
        if ci in (0, 4) or (ci + 1) % max(1, nchunk // 10) == 0 or ci == nchunk - 1:
            el = time.time() - t0
            print('    chunk %d/%d  %.0f patch-samples/s  eta %.0f min'
                  % (ci + 1, nchunk, (ci + 1) * batch * nsamp / el, el / (ci + 1) * (nchunk - ci - 1) / 60),
                  flush=True)
    mean = S1 / W[..., None]
    std = np.sqrt(np.maximum(S2 / W[..., None] - mean ** 2, 0))
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32), label=lab, input=inp,
                row=row, band=np.array([i0, i0 + p]), nsamp=nsamp)


def recon(counts, air, geom, parallel):
    """counts (nview, nch, C) -> (C, npix, npix): -log(counts / air) per bin, then FBP."""
    li = -np.log(np.maximum(counts, LI_FLOOR) / np.asarray(air, np.float64)[None, None, :])
    f = geom.fbp_parallel if parallel else geom.fbp
    return np.stack([f(np.ascontiguousarray(li[:, :, b].T, np.float64), window=FBP_WINDOW)
                     for b in range(counts.shape[-1])])


def floored(counts):
    """Fraction of sinogram values at or below the -log floor, per bin."""
    return (np.asarray(counts) <= LI_FLOOR).mean((0, 1))


def run_recon(a, ds, get_sampler, wd, labels, title):
    man = ds.man
    if man['nch'] != 1854:
        sys.exit('image domain needs the full detector (generated with --crop 1854); '
                 'this arm has %d channels' % man['nch'])
    geom = Geometry(nview=man['nview'], nch=man['nch'], npix=man['npix'])
    parallel = 'parallel' in man.get('geometry', '')
    air = man.get('air_counts')
    row = a.row if a.row is not None else man['nrow'] // 2
    phs = a.phantoms if a.phantoms is not None else list(range(len(ds.files)))
    vis = a.vis_phantom if a.vis_phantom is not None else phs[0]
    print('\nimage domain: %d test phantom(s), row %d, %s-beam FBP, %d samples per patch'
          % (len(phs), row, 'parallel' if parallel else 'fan', a.nsamp_recon), flush=True)
    per = []
    for fi in phs:
        f = os.path.join(wd, 'slice_ph%03d_row%02d.npz' % (fi, row))
        sl = None
        if os.path.exists(f) and not a.resample:
            z = dict(np.load(f))
            if int(z['nsamp']) == a.nsamp_recon:
                sl = z; print('  phantom %d: reusing %s' % (fi, f))
        if sl is None:
            sl = slice_posterior(ds, fi, get_sampler(), row, man['stride'], a.nsamp_recon, a.batch)
            np.savez(f, **sl)
        air_f = np.asarray(air) if air is not None else sl['label'].max((0, 1))
        Rt = recon(sl['label'], air_f, geom, parallel)
        Rm = recon(sl['mean'], air_f, geom, parallel)
        Rx = recon(sl['input'], air_f, geom, parallel)
        np.savez(os.path.join(wd, 'recon_ph%03d_row%02d.npz' % (fi, row)),
                 label=Rt.astype(np.float32), mean=Rm.astype(np.float32), input=Rx.astype(np.float32))
        L = Rt.max((1, 2)) - Rt.min((1, 2))
        fl = {k: floored(sl[k]) for k in ('label', 'mean', 'input')}
        print('  phantom %d: sinogram values at the %.1f-count floor (worst bin): '
              'label %.3f%%  mean %.3f%%  input %.3f%%'
              % (fi, LI_FLOOR, *(100 * fl[k].max() for k in ('label', 'mean', 'input'))))
        per.append(dict(phantom=fi, data_range=L.tolist(),
                        floored={k: v.tolist() for k, v in fl.items()},
                        ssim=ssim_windowed(Rm, Rt, L).tolist(),
                        ssim_input=ssim_windowed(Rx, Rt, L).tolist(),
                        rmse=np.sqrt(((Rm - Rt) ** 2).mean((1, 2))).tolist()))
        if fi == vis:
            figure(sl, a.vis_bins, a.vis_ch, labels, fi, title, a.nsamp_recon)

    S = np.array([q['ssim'] for q in per]); Sx = np.array([q['ssim_input'] for q in per])
    print('\n--- IMAGE domain: FBP slice (row %d), per-bin windowed SSIM, mean over %d phantom(s) ---'
          % (row, len(per)))
    print('  %-12s %9s %9s' % ('bin', 'model', 'input'))
    for b in range(S.shape[1]):
        print('  %-12s %9.4f %9.4f' % (labels[b], S[:, b].mean(), Sx[:, b].mean()))
    print('  %-12s %9.4f %9.4f' % ('all bins', S.mean(), Sx.mean()))
    return dict(row=row, nsamp=a.nsamp_recon, fbp='parallel' if parallel else 'fan',
                fbp_window=FBP_WINDOW, li_floor=LI_FLOOR, phantoms=per, ssim_per_bin=S.mean(0).tolist(),
                ssim=float(S.mean()), ssim_input_per_bin=Sx.mean(0).tolist(),
                ssim_input=float(Sx.mean()))


# ---------------------------------------------------------------- C: figure
def figure(sl, bins, ch, labels, fi, title, nsamp):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nch = sl['label'].shape[1]
    c0, c1 = ch if ch else (nch // 2 - 200, nch // 2 + 200)
    heads = ['distorted input', 'clean label', 'posterior mean (%d samples)' % nsamp,
             'error: mean - label', 'posterior std']
    fig, ax = plt.subplots(len(bins), 5, figsize=(18, 3.1 * len(bins) + 0.9), squeeze=False)
    for r, b in enumerate(bins):
        A = {k: sl[k][:, c0:c1, b] for k in ('input', 'label', 'mean', 'std')}
        err = A['mean'] - A['label']
        lo, hi = np.percentile(A['label'], [1, 99])
        m = max(np.percentile(np.abs(err), 99), np.percentile(A['std'], 99), 1e-6)
        panels = [(A['input'], 'gray', lo, hi), (A['label'], 'gray', lo, hi),
                  (A['mean'], 'gray', lo, hi), (err, 'RdBu_r', -m, m), (A['std'], 'magma', 0, m)]
        for c, (img, cm, vmin, vmax) in enumerate(panels):
            im = ax[r, c].imshow(img, cmap=cm, vmin=vmin, vmax=vmax, aspect='auto',
                                 interpolation='nearest', extent=[c0, c1, img.shape[0], 0])
            fig.colorbar(im, ax=ax[r, c], fraction=0.046, pad=0.02, label='counts')
            if r == 0: ax[r, c].set_title(heads[c], fontsize=10)
            if c == 0: ax[r, c].set_ylabel('%s\nview' % labels[b])
            if r == len(bins) - 1: ax[r, c].set_xlabel('detector channel')
        ax[r, 3].text(0.02, 0.97, 'RMSE %.2f' % np.sqrt((err ** 2).mean()), transform=ax[r, 3].transAxes,
                      va='top', fontsize=8, bbox=dict(fc='white', alpha=0.7, lw=0))
    fig.suptitle('%s -- test phantom %d, sinogram of detector row %d, channels %d-%d'
                 % (title, fi, int(sl['row']), c0, c1 - 1))
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, 'sino_error_%s_ph%03d.png' % (title.replace(' ', '_'), fi))
    fig.savefig(p, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('  wrote', p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train_arm', default='baseline3d_pu_matched', help='arm the model was trained on')
    ap.add_argument('--arm', default=None, help='arm to evaluate on (default: the training arm)')
    ap.add_argument('--ckpt', default=None, help='default: outputs/edm_<train_arm>_Y/ckpt.pt')
    ap.add_argument('--parts', nargs='+', default=['patch', 'recon'], choices=['patch', 'recon'])
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--npatch', type=int, default=256)
    ap.add_argument('--nsamp', type=int, default=256, help='patch domain; 256 = coverage.py')
    ap.add_argument('--nsamp_recon', type=int, default=16)
    ap.add_argument('--steps', type=int, default=18)
    ap.add_argument('--batch', type=int, default=2048, help='patches per model call (image domain)')
    ap.add_argument('--row', type=int, default=None, help='detector row = slice (default nrow//2)')
    ap.add_argument('--phantoms', type=int, nargs='+', default=None, help='test phantom indices')
    ap.add_argument('--vis_phantom', type=int, default=None)
    ap.add_argument('--vis_bins', type=int, nargs='+', default=[0, 4, 8])
    ap.add_argument('--vis_ch', type=int, nargs=2, default=None, help='channel range to show')
    ap.add_argument('--device', default=None)
    ap.add_argument('--resample', action='store_true', help='ignore saved samples, redraw')
    a = ap.parse_args()
    if a.nsamp_recon >= 1000:
        sys.exit('--nsamp_recon must be < 1000 (seed layout)')

    arm = a.arm or a.train_arm
    a.arm = arm
    ckpt = a.ckpt or os.path.join(OUT, 'edm_%s_Y' % a.train_arm, 'ckpt.pt')
    tag = '%s_Y_on_%s' % (a.train_arm, arm)
    wd = os.path.join(OUT, 'ssim_' + tag); os.makedirs(wd, exist_ok=True)
    ds = DiffusionPatches(arm, 'test', a.patch, 'Y')
    labels = bin_labels(ds.man)

    box = {}
    def get_sampler():
        if 's' not in box:
            s = Sampler(ckpt, a.device, a.steps)
            if s.trained_on != a.train_arm:
                sys.exit('%s was trained on %s, not --train_arm %s' % (ckpt, s.trained_on, a.train_arm))
            print('checkpoint: arm=%s it=%d device=%s' % (s.trained_on, s.it, s.device), flush=True)
            box['s'] = s
        return box['s']

    fp = os.path.join(OUT, 'ssim_%s.json' % tag)
    res = json.load(open(fp)) if os.path.exists(fp) else {}
    res.update(train_arm=a.train_arm, arm=arm, bins=labels)
    if 'patch' in a.parts:
        res['patch'] = run_patch(a, ds, get_sampler, wd, labels, a.train_arm)
    if 'recon' in a.parts:
        res['recon'] = run_recon(a, ds, get_sampler, wd, labels, tag)
    with open(fp, 'w') as f:
        json.dump(res, f, indent=2)
    print('\nwrote', fp)


if __name__ == '__main__':
    main()

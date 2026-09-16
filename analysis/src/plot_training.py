"""Training curves for the diffusion model and the WGAN baseline.

    python3 plot_training.py                    # reads ../logs and ../outputs
    python3 plot_training.py --out ../figures/training_curves.png --split_dir ../figures/training
    python3 plot_training.py --logs ~/somewhere/logs

--out is one overview figure (all curves); --split_dir also writes one figure per
curve (NN_<name>.png) and training_summary.txt.

Sources, per model, in order of preference:
  1  outputs/<edm|wgan>_<arm>_Y/train_log.csv   (trainlog.py; runs started or
     resumed after 2026-09-14)
  2  logs/<edm|wgan>_<jobid>.out                 (the SLURM logs; every run)
Both are merged, so a run whose early part is only in the .out logs and whose
later part is in the CSV still gives one curve.

Resumed jobs overlap: a job killed at iteration 25,000 and resumed from the
20,000 checkpoint logs 20,200-25,000 twice. The LATER job wins -- it is the run
that continued. Vertical dotted lines mark where each job took over.

Every value is the logged mean over --log_every (200) iterations, not one step --
EXCEPT the WGAN critic loss from .out logs: train.py prints lD.item(), the last
critic step before the log line, so that curve is single steps and noisier. The
CSV logs (D_mean) carry the true interval mean. The thick line is a rolling median
over 10 log lines (2,000 iterations), for the eye only.

The diffusion loss (EDM denoising loss, averaged over noise levels) and the WGAN
terms are different objectives in different units: they are drawn in separate
panels and must not be compared by value.
"""
import argparse, csv, glob, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANAL = os.path.dirname(HERE)

EDM_RE = re.compile(r'^it\s+(\d+)\s+loss\s+([-\d.eE+]+)\s+([\d.]+) it/s')
WGAN_RE = re.compile(r'^it\s+(\d+)\s+D\s+([-+\d.eE]+)\s+G\s+([-+\d.eE]+)\s+mse\s+([-+\d.eE]+)\s+'
                     r'rmae\s+([-+\d.eE]+)\s+adv\s+([-+\d.eE]+)\s+([\d.]+) it/s')

# (model, key, title, log y, y label, file name)
PANELS = [
    ('edm', 'loss', 'Diffusion (EDM): denoising loss', True, 'loss', 'diffusion_loss'),
    ('edm', 'it_per_s', 'Diffusion (EDM): training speed', False, 'iterations / s', 'diffusion_speed'),
    ('wgan', 'G_mse', 'WGAN: MSE term (normalised units)', True, 'MSE', 'wgan_mse'),
    ('wgan', 'G_rmae', 'WGAN: relative MAE term', True, 'relative MAE', 'wgan_rmae'),
    ('wgan', 'G_total', 'WGAN: generator loss, weighted total', True, 'loss', 'wgan_generator_total'),
    ('wgan', 'G_adv', 'WGAN: adversarial term  -D(G(x))', False, 'value', 'wgan_adversarial'),
    ('wgan', 'D', 'WGAN: critic loss (incl. gradient penalty)', False, 'value', 'wgan_critic'),
    ('wgan', 'it_per_s', 'WGAN: training speed (loader fix between jobs)', False, 'iterations / s',
     'wgan_speed'),
]
COLOUR = {'edm': 'C0', 'wgan': 'C3'}


def parse_out(path, model):
    """-> {it: row dict}, the job id, and the iteration it resumed from (0 if fresh)."""
    rows, resumed = {}, 0
    job = re.search(r'_(\d+)\.out$', path)
    job = job.group(1) if job else os.path.basename(path)
    for line in open(path, errors='replace'):
        m = re.search(r'resumed from .* at iteration (\d+)', line)
        if m:
            resumed = int(m.group(1)); continue
        if model == 'edm':
            m = EDM_RE.match(line)
            if m:
                rows[int(m.group(1))] = dict(loss=float(m.group(2)), it_per_s=float(m.group(3)), job=job)
        else:
            m = WGAN_RE.match(line)
            if m:
                g = [float(v) for v in m.groups()[1:]]
                # the .out line's D is lD.item(): the LAST critic step, not a mean
                rows[int(m.group(1))] = dict(D=g[0], D_src='last', G_total=g[1], G_mse=g[2],
                                             G_rmae=g[3], G_adv=g[4], it_per_s=g[5], job=job)
    return rows, job, resumed


def parse_csv(path, model):
    rows = {}
    for r in csv.DictReader(open(path)):
        d = {k: (float(v) if v not in ('', None) and k != 'job' else v) for k, v in r.items() if k != 'it'}
        if model == 'wgan':
            d['D'] = d.pop('D_mean', d.get('D_last'))      # the interval mean when available
            d['D_src'] = 'mean' if 'D_mean' in r else 'last'
        rows[int(r['it'])] = d
    return rows


def _jobnum(p):
    m = re.search(r'_(\d+)\.out$', p)
    return int(m.group(1)) if m else 0


def load(model, logs, outputs, arm):
    """Merge every source for one model. Jobs are applied in job-id order, so a
    later (resumed) job overwrites the iterations it repeated."""
    merged, starts = {}, []
    outs = sorted(glob.glob(os.path.join(logs, '%s_*.out' % model)), key=_jobnum)
    for p in outs:
        rows, job, resumed = parse_out(p, model)
        if rows:
            merged.update(rows)
            starts.append((min(rows), job, resumed))
    c = os.path.join(outputs, '%s_%s_Y' % (model, arm), 'train_log.csv')
    if os.path.exists(c):
        merged.update(parse_csv(c, model))
    its = np.array(sorted(merged))
    return its, merged, starts, outs, os.path.exists(c)


def smooth(y, k=10):
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode='edge')
    return np.array([np.median(pad[i:i + k]) for i in range(len(y))])


def draw(ax, data, model, key, title, logy, ylab, plain=False):
    """plain: no resume markers and no bracketed detail in the title."""
    its, rows, starts = data[model][:3]
    if plain and title.endswith(')'):
        title = title.rsplit(' (', 1)[0]
    y = np.array([rows[i].get(key, np.nan) for i in its], float)
    ok = np.isfinite(y)
    if not ok.any():
        ax.set_visible(False); return False
    c = COLOUR[model]
    lab = 'mean over 200 iterations'
    if key == 'D':                         # .out logs print the last critic step only
        src = {rows[i].get('D_src') for i in its[ok]}
        lab = ('last critic step, every 200 iterations' if src == {'last'} else
               'last step (.out) / 200-iteration mean (CSV)' if 'last' in src else lab)
    ax.plot(its[ok], y[ok], '.', ms=2, color=c, alpha=0.35, label=lab)
    ax.plot(its[ok], smooth(y[ok]), '-', lw=1.6, color=c, label='rolling median of 10')
    for k, (first, job, resumed) in enumerate([] if plain else starts[1:]):
        ax.axvline(first, color='0.4', ls=':', lw=1,
                   label='resumed job takes over' if k == 0 else None)
    if logy and (y[ok] > 0).all():
        ax.set_yscale('log')
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlabel('iteration'); ax.set_ylabel(ylab)
    return True


def summary(data, wgan_iters, plain=False):
    out = []
    its, rows, starts, outs, has_csv = data['edm']
    if len(its):
        jobs = '' if plain else ', %d job(s): %s' % (len(starts), ', '.join(j for _, j, _ in starts))
        out.append('Diffusion (EDM)\n  %d iterations%s\n'
                   '  first logged loss %.4f, final %.5f\n  speed %.1f it/s'
                   % (its[-1], jobs, rows[its[0]]['loss'], rows[its[-1]]['loss'], rows[its[-1]]['it_per_s']))
    its, rows, starts, outs, has_csv = data['wgan']
    if len(its):
        rate = rows[its[-1]]['it_per_s']
        left = max(wgan_iters - its[-1], 0)
        jobs = '' if plain else ', %d job(s): %s' % (len(starts), ', '.join(
            '%s%s' % (j, ' (resumed from %d)' % r if r else '') for _, j, r in starts))
        out.append('WGAN-GP\n  %d / %d iterations (%.1f%%)%s\n'
                   '  speed %.1f it/s -> ~%.0f h left'
                   % (its[-1], wgan_iters, 100 * its[-1] / wgan_iters, jobs, rate, left / rate / 3600))
    return '\n\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--logs', default=os.path.join(ANAL, 'logs'))
    ap.add_argument('--outputs', default=os.path.join(ANAL, 'outputs'))
    ap.add_argument('--arm', default='baseline3d_pu_matched')
    ap.add_argument('--out', default=os.path.join(ANAL, 'figures', 'training_curves.png'))
    ap.add_argument('--split_dir', default=None, help='also write one figure per curve here')
    ap.add_argument('--plain', action='store_true',
                    help='for reports: no resume markers, legends, job ids or bracketed details')
    ap.add_argument('--wgan_iters', type=int, default=1086750,
                    help='planned WGAN iterations (40 epochs), for the progress line')
    a = ap.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {m: load(m, a.logs, a.outputs, a.arm) for m in ('edm', 'wgan')}
    for m, nm in (('edm', 'diffusion'), ('wgan', 'WGAN')):
        its, rows, starts, outs, has_csv = data[m]
        print('%-9s %d log lines, iterations %s, from %d .out file(s)%s'
              % (nm, len(its), '%d-%d' % (its[0], its[-1]) if len(its) else 'none', len(outs),
                 ' + train_log.csv' if has_csv else ''))
        for first, job, resumed in starts:
            print('            job %s: first line at it %d%s' % (job, first,
                  ', resumed from checkpoint %d' % resumed if resumed else ''))
    text = summary(data, a.wgan_iters, a.plain)
    note = ('point = mean over 200 iterations (WGAN critic from .out logs: last step only); '
            'line = rolling median of 10; dotted = a resumed job takes over')
    if a.plain:
        note = 'points = logged values, line = smoothed'

    # overview: all curves on one page, summary text in the third slot
    fig, ax = plt.subplots(3, 3, figsize=(17, 12))
    ax = ax.ravel()
    slots = [0, 1, 3, 4, 5, 6, 7, 8]
    for s, (m, key, title, logy, ylab, _) in zip(slots, PANELS):
        draw(ax[s], data, m, key, title, logy, ylab, a.plain)
    ax[2].axis('off')
    ax[2].text(0, 0.95, text, va='top', family='monospace', fontsize=9.5, transform=ax[2].transAxes)
    fig.suptitle('Training curves -- %s' % note if a.plain else
                 'Training curves, arm %s -- %s' % (a.arm, note), fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=110, bbox_inches='tight'); plt.close(fig)
    print('wrote', a.out)

    if a.split_dir:
        os.makedirs(a.split_dir, exist_ok=True)
        for n, (m, key, title, logy, ylab, fname) in enumerate(PANELS, 1):
            fig, ax1 = plt.subplots(figsize=(8, 4.8))
            if draw(ax1, data, m, key, title, logy, ylab, a.plain):
                if not a.plain:
                    ax1.legend(fontsize=8, loc='best')
                    fig.text(0.99, 0.005, 'arm %s' % a.arm, ha='right', va='bottom', fontsize=7, color='0.4')
                fig.tight_layout()
                p = os.path.join(a.split_dir, '%02d_%s.png' % (n, fname))
                fig.savefig(p, dpi=130, bbox_inches='tight')
                print('wrote', p)
            plt.close(fig)
        with open(os.path.join(a.split_dir, 'training_summary.txt'), 'w') as f:
            f.write(text + '\n')
        print('wrote', os.path.join(a.split_dir, 'training_summary.txt'))


if __name__ == '__main__':
    main()

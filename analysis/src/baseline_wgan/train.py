"""Train the Morovati et al. 2025 baseline (residual WGAN-GP).

    python3 train.py --arm baseline3d --iters 200000 --batch 64

Uses the SAME data loader and the SAME frozen normalisation as the diffusion arm
(diffusion/norm_stats.py), so the two are directly comparable. Target is the
noise-free ideal bin counts.

Their reported settings: Adam, lr 1e-4, 40 epochs to MSE convergence, 16x16x9
patches. Batch size is not stated in the paper; 64 is used here.

The perceptual term needs a ViT trained on PCCT data -- they explicitly do NOT
use an ImageNet-pretrained network. There is no checkpoint to download, so either
train one and pass --vit_ckpt, or set --lam_perc 0 and say so in the write-up.
"""
import argparse, json, os, time
import numpy as np
import torch

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../diffusion')
from paths import OUT
from data import DiffusionPatches
from models import Generator, Discriminator, SmallViT
from losses import critic_loss, generator_loss
from trainlog import CsvLog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='baseline3d')
    ap.add_argument('--target', default='Y', choices=['Y', 'V'])
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--iters', type=int, default=200_000)
    ap.add_argument('--epochs', type=float, default=None,
                    help='overrides --iters: passes over the train patches. Morovati et al. '
                         'train 40 epochs, which on our data is ~1.09M iterations at batch 64.')
    ap.add_argument('--preload', action='store_true',
                    help='load every train phantom into RAM once (~0.77 GB each) instead of '
                         're-decompressing a file every 64 batches -- measured at ~90%% of the '
                         'WGAN iteration time. Refuses to start if the memory is not there.')
    ap.add_argument('--resume', action='store_true',
                    help='continue from out/ckpt.pt if it exists (no-op otherwise), so a '
                         'run longer than the SLURM time limit can be resubmitted')
    ap.add_argument('--lr', type=float, default=1e-4)          # their alpha
    ap.add_argument('--n_critic', type=int, default=5)         # WGAN-GP default
    ap.add_argument('--lam_gp', type=float, default=10.0)
    ap.add_argument('--lam_adv', type=float, default=1.0)
    ap.add_argument('--lam_mse', type=float, default=1000.0)
    ap.add_argument('--lam_rmae', type=float, default=1000.0)
    ap.add_argument('--lam_perc', type=float, default=0.0,
                    help='10.0 in the paper, but needs a PCCT-trained ViT; 0 disables it')
    ap.add_argument('--vit_ckpt', default=None)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--rmae_floor', type=float, default=1.0,
                    help='RMAE denominator floor, in COUNTS (see losses.py)')
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    ap.add_argument('--log_every', type=int, default=200)
    ap.add_argument('--ckpt_every', type=int, default=10_000)
    a = ap.parse_args()

    out = a.out or os.path.join(OUT, 'wgan_%s_%s' % (a.arm, a.target))
    os.makedirs(out, exist_ok=True)
    ds = DiffusionPatches(a.arm, 'train', a.patch, a.target, preload=a.preload)
    ch = ds.tgt_ch
    if a.epochs is not None:
        n_patch = ds.man['patches_per_projection'] * ds.man['nview'] * len(ds.files)
        a.iters = int(np.ceil(a.epochs * n_patch / a.batch))
        print('%g epochs over %d train patches at batch %d -> %d iterations'
              % (a.epochs, n_patch, a.batch, a.iters), flush=True)
    print('arm=%s target=%s ch=%d patch=%d' % (a.arm, a.target, ch, a.patch), flush=True)

    # map standardised log1p values back to physical units for the RMAE term
    tm = torch.tensor(ds.tm, device=a.device).view(1, -1, 1, 1)
    ts = torch.tensor(ds.ts, device=a.device).view(1, -1, 1, 1)
    if a.target == 'Y':
        to_phys = lambda z: torch.expm1((z * ts + tm).clamp(0.0, 16.0))   # counts, <= ~1e7
    else:
        to_phys = lambda z: z * ts + tm                                     # cm
    G = Generator(ch=ch, base=a.base).to(a.device)
    D = Discriminator(ch=ch, base=a.base, patch=a.patch).to(a.device)
    vit = None
    if a.lam_perc > 0:
        vit = SmallViT(ch=ch, img=a.patch).to(a.device)
        if a.vit_ckpt:
            vit.load_state_dict(torch.load(a.vit_ckpt, map_location=a.device))
        else:
            print('WARNING: --lam_perc > 0 with an UNTRAINED ViT. Its features are '
                  'random, so the perceptual term is noise. Train it or set --lam_perc 0.',
                  flush=True)
        for p in vit.parameters():
            p.requires_grad_(False)
        vit.eval()
    print('G %.2fM  D %.2fM' % (sum(p.numel() for p in G.parameters()) / 1e6,
                                sum(p.numel() for p in D.parameters()) / 1e6), flush=True)

    oG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.5, 0.9))
    oD = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.5, 0.9))
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(vars(a), f, indent=2)

    ck = os.path.join(out, 'ckpt.pt')
    start = 1
    if a.resume and os.path.exists(ck):
        c = torch.load(ck, map_location=a.device)
        G.load_state_dict(c['G']); D.load_state_dict(c['D'])
        if 'oG' in c:
            oG.load_state_dict(c['oG']); oD.load_state_dict(c['oD'])
        start = c['it'] + 1
        print('resumed from %s at iteration %d' % (ck, c['it']), flush=True)

    def save(it):
        tmp = ck + '.tmp'                     # atomic: a kill mid-save cannot corrupt it
        torch.save(dict(G=G.state_dict(), D=D.state_dict(), oG=oG.state_dict(),
                        oD=oD.state_dict(), it=it, cfg=vars(a), ch=ch), tmp)
        os.replace(tmp, ck)

    # curves for after training: out/train_log.csv (trainlog.py). Columns come from the
    # generator's loss terms, so the file is opened at the first log line.
    job = os.environ.get('SLURM_JOB_ID', '')
    log = None

    t0 = time.time(); hist = []; dh = []
    for it in range(start, a.iters + 1):
        for _ in range(a.n_critic):
            x, y = ds.batch(a.batch)
            xb = torch.from_numpy(x).to(a.device); yb = torch.from_numpy(y).to(a.device)
            with torch.no_grad():
                fake = G(xb)
            lD = critic_loss(D, yb, fake, lam_gp=a.lam_gp)
            oD.zero_grad(set_to_none=True); lD.backward(); oD.step()

        x, y = ds.batch(a.batch)
        xb = torch.from_numpy(x).to(a.device); yb = torch.from_numpy(y).to(a.device)
        lG, terms = generator_loss(D, G(xb), yb, lam_adv=a.lam_adv, lam_mse=a.lam_mse,
                                   lam_rmae=a.lam_rmae, lam_perc=a.lam_perc,
                                   eps=a.eps, vit=vit, to_phys=to_phys,
                                   rmae_floor=a.rmae_floor)
        oG.zero_grad(set_to_none=True); lG.backward(); oG.step()

        hist.append(terms); dh.append(lD.item())       # lD: the last critic step
        if it % a.log_every == 0:
            m = {k: float(np.mean([h[k] for h in hist])) for k in terms}
            rate = (it - start + 1) / (time.time() - t0)
            print('it %7d  D %+8.3f  G %8.3f  mse %.5f  rmae %.4f  adv %+.3f  %.1f it/s'
                  % (it, lD.item(), m['total'], m['mse'], m['rmae'], m['adv'], rate), flush=True)
            if log is None:
                log = CsvLog(os.path.join(out, 'train_log.csv'),
                             ['it', 'D_mean', 'D_last'] + ['G_' + k for k in terms]
                             + ['it_per_s', 'elapsed_s', 'job'], keep_upto=start - 1)
                if log.kept:
                    print('train_log.csv: kept %d rows up to iteration %d' % (log.kept, start - 1),
                          flush=True)
            log.row(it=it, D_mean=float(np.mean(dh)), D_last=lD.item(),
                    **{'G_' + k: v for k, v in m.items()}, it_per_s=rate,
                    elapsed_s=time.time() - t0, job=job)
            hist, dh = [], []                  # was never cleared: ~1M dicts by the end
        if it % a.ckpt_every == 0 or it == a.iters:
            save(it)
    print('done ->', out)


if __name__ == '__main__':
    main()

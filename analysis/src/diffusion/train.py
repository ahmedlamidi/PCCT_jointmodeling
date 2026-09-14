"""Train the conditional EDM.

    python3 train.py --arm baseline3d --target V --iters 200000 --batch 128

Defaults follow Karras et al. 2022 (P_mean=-1.2, P_std=1.2, sigma_data=1.0 by
construction since norm_stats standardises the target to unit variance).

EMA of the weights is what you sample from -- EDM's results depend on it.
"""
import argparse, json, os, time
import numpy as np
import torch

import sys; sys.path.insert(0, '..')
from paths import OUT
from data import DiffusionPatches
from unet import UNet
from edm import EDMPrecond, edm_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='baseline3d', help='dataset arm to TRAIN on')
    ap.add_argument('--target', default='Y', choices=['Y', 'V'],
                    help="'Y' = ideal bin counts, which is what Morovati et al. predict "
                         "(bin counts in, bin counts out). 'V' = material line integrals.")
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--iters', type=int, default=200_000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--base', type=int, default=96)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--ema', type=float, default=0.9999)
    ap.add_argument('--sigma_data', type=float, default=1.0)
    ap.add_argument('--P_mean', type=float, default=-1.2)
    ap.add_argument('--P_std', type=float, default=1.2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    ap.add_argument('--log_every', type=int, default=200)
    ap.add_argument('--ckpt_every', type=int, default=10_000)
    ap.add_argument('--preload', action='store_true',
                    help='load every train phantom into RAM once (~0.77 GB each) instead of '
                         're-decompressing a file every 64 batches -- measured at ~90%% of the '
                         'WGAN iteration time. Refuses to start if the memory is not there.')
    ap.add_argument('--resume', action='store_true',
                    help='continue from out/ckpt.pt if it exists (no-op otherwise)')
    a = ap.parse_args()

    out = a.out or os.path.join(OUT, 'edm_%s_%s' % (a.arm, a.target))
    os.makedirs(out, exist_ok=True)
    ds = DiffusionPatches(a.arm, 'train', a.patch, a.target, preload=a.preload)
    print('arm=%s target=%s  cond_ch=%d tgt_ch=%d  patch=%d'
          % (a.arm, a.target, ds.cond_ch, ds.tgt_ch, a.patch), flush=True)

    model = UNet(ds.tgt_ch + ds.cond_ch, ds.tgt_ch, base=a.base, dropout=a.dropout)
    net = EDMPrecond(model, sigma_data=a.sigma_data).to(a.device)
    ema = EDMPrecond(UNet(ds.tgt_ch + ds.cond_ch, ds.tgt_ch, base=a.base),
                     sigma_data=a.sigma_data).to(a.device)
    ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    nparam = sum(p.numel() for p in net.parameters())
    print('params %.2f M' % (nparam / 1e6), flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.9, 0.999))
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(vars(a) | dict(params=nparam), f, indent=2)

    ck = os.path.join(out, 'ckpt.pt')
    start = 1
    if a.resume and os.path.exists(ck):
        c = torch.load(ck, map_location=a.device)
        net.load_state_dict(c['net']); ema.load_state_dict(c['ema'])
        if 'opt' in c:
            opt.load_state_dict(c['opt'])
        start = c['it'] + 1
        print('resumed from %s at iteration %d' % (ck, c['it']), flush=True)

    def save(it):
        tmp = ck + '.tmp'                     # atomic: a kill mid-save cannot corrupt it
        torch.save(dict(ema=ema.state_dict(), net=net.state_dict(), opt=opt.state_dict(),
                        it=it, cfg=vars(a), cond_ch=ds.cond_ch, tgt_ch=ds.tgt_ch), tmp)
        os.replace(tmp, ck)

    t0 = time.time(); run = []
    for it in range(start, a.iters + 1):
        for g in opt.param_groups:                      # linear warmup
            g['lr'] = a.lr * min(it / max(a.warmup, 1), 1.0)
        x, y = ds.batch(a.batch)
        cond = torch.from_numpy(x).to(a.device)
        tgt = torch.from_numpy(y).to(a.device)
        loss = edm_loss(net, tgt, cond, P_mean=a.P_mean, P_std=a.P_std)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        with torch.no_grad():                            # EMA
            d = min(a.ema, (it + 1) / (it + 10))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(d).add_(pn.detach(), alpha=1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)

        run.append(loss.item())
        if it % a.log_every == 0:
            print('it %7d  loss %.5f  %.1f it/s' % (it, np.mean(run[-a.log_every:]),
                  (it - start + 1) / (time.time() - t0)), flush=True)
        if it % a.ckpt_every == 0 or it == a.iters:
            save(it)
    print('done ->', out)


if __name__ == '__main__':
    main()

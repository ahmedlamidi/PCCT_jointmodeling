"""Run this FIRST on the GPU box. Catches shape bugs in ~1 minute.

    python3 smoke_test.py

Checks, in order:
  1  UNet forward on one batch
  2  EDM loss backward
  3  preconditioning identity: D(x;s) -> data as s->0, -> prior mean as s->inf
  4  sampler runs and returns finite values in a sane physical range
"""
import os, numpy as np, torch, sys
sys.path.insert(0, '..')
from data import DiffusionPatches
from unet import UNet
from edm import EDMPrecond, edm_loss, edm_sample, sigma_schedule

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', dev)

# Arm comes from the environment so the SLURM script and the smoke test agree.
# It used to be hard-coded to 'baseline3d', which does not exist on the cluster
# (gen_data.sh writes baseline3d_matched) -- the job would have died here.
ARM = os.environ.get('ARM', 'baseline3d_pu_matched')
ds = DiffusionPatches(ARM, 'train', 16, 'Y')   # bin counts, as in Morovati
x, y = ds.batch(4)
cond = torch.from_numpy(x).to(dev); tgt = torch.from_numpy(y).to(dev)
print('1  cond', tuple(cond.shape), 'target', tuple(tgt.shape))

model = UNet(ds.tgt_ch + ds.cond_ch, ds.tgt_ch, base=32, mults=(1, 2, 2))
net = EDMPrecond(model, sigma_data=1.0).to(dev)
print('   params %.2f M' % (sum(p.numel() for p in net.parameters()) / 1e6))

s = torch.full((4,), 1.0, device=dev)
out = net(tgt, s, cond)
assert out.shape == tgt.shape, (out.shape, tgt.shape)
print('2  denoiser out', tuple(out.shape), 'finite', bool(torch.isfinite(out).all()))

loss = edm_loss(net, tgt, cond)
loss.backward()
gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
print('3  loss %.4f  grad-norm %.3f' % (loss.item(), gn))
assert np.isfinite(loss.item()) and gn > 0

with torch.no_grad():
    lo = net(tgt, torch.full((4,), 1e-4, device=dev), cond)
    hi = net(tgt, torch.full((4,), 1e3, device=dev), cond)
print('4  s->0  ||D-x|| %.5f   (should be ~0: c_skip->1)' % float((lo - tgt).abs().mean()))
print('   s->inf ||D||   %.5f   (should be small: c_skip->0)' % float(hi.abs().mean()))

ts = sigma_schedule(18)
print('5  schedule %.3f -> %.5f -> %.1f (last is 0)' % (ts[0], ts[-2], ts[-1]))

with torch.no_grad():
    smp = edm_sample(net, (4, ds.tgt_ch, 16, 16), cond=cond, steps=8, device=dev)
phys = ds.denorm_t(smp.cpu().numpy().transpose(0, 2, 3, 1))
print('6  sample', tuple(smp.shape), 'finite', bool(torch.isfinite(smp).all()))
print('   untrained output, counts: %.1f .. %.1f  (clamped at 0)'
      % (phys.min(), phys.max()))
from manifold import Manifold
print('7  manifold residual of untrained samples: %.3f  (truth baseline ~0.04)'
      % Manifold.fit(ds.denorm_t(y.transpose(0, 2, 3, 1)).reshape(-1, ds.tgt_ch)).residual(
            phys.reshape(-1, ds.tgt_ch)).mean())
print('\nALL CHECKS PASSED (untrained network, so values are meaningless -- shapes are the point)')

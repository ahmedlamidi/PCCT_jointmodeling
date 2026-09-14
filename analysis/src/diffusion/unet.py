"""Small UNet for 16x16 patches.

The standard EDM UNets (SongUNet / DhariwalUNet) assume >=32x32 and are far too
deep here -- a 16x16 patch survives only two downsamples. This is a compact
version with the pieces that matter: GroupNorm + SiLU residual blocks, Fourier
noise-level embedding, self-attention at the bottleneck.

Receptive field note: the deterministic distortion has 3x3 support (PcTK's
covariance is 9 pixels), so depth here is about capacity, not reach.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class FourierEmb(nn.Module):
    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.register_buffer('freqs', torch.randn(dim // 2) * scale)

    def forward(self, t):
        x = t.float().ger((2 * np.pi * self.freqs).to(t.device))
        return torch.cat([x.cos(), x.sin()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb, groups=8, dropout=0.0):
        super().__init__()
        g1 = min(groups, cin); g2 = min(groups, cout)
        self.n1 = nn.GroupNorm(g1, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb, cout * 2)
        self.n2 = nn.GroupNorm(g2, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(Fn.silu(self.n1(x)))
        scale, shift = self.emb(Fn.silu(e))[:, :, None, None].chunk(2, dim=1)
        h = Fn.silu(self.n2(h) * (1 + scale) + shift)
        return self.c2(self.drop(h)) + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, groups=8):
        super().__init__()
        self.n = nn.GroupNorm(min(groups, c), c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).unbind(1)
        a = torch.softmax(q.transpose(1, 2) @ k / np.sqrt(C), dim=-1)
        return x + self.out((v @ a.transpose(1, 2)).reshape(B, C, H, W))


class UNet(nn.Module):
    """in_ch = target channels + condition channels; out_ch = target channels."""

    def __init__(self, in_ch, out_ch, base=96, mults=(1, 2, 2), emb=256,
                 attn_at=(2,), dropout=0.0):
        super().__init__()
        self.temb = nn.Sequential(FourierEmb(emb), nn.Linear(emb, emb),
                                  nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv2d(in_ch, base, 3, padding=1)
        chans = [base * m for m in mults]

        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        c = base; skips = [base]
        for i, ch in enumerate(chans):
            blocks = nn.ModuleList([ResBlock(c, ch, emb, dropout=dropout),
                                    ResBlock(ch, ch, emb, dropout=dropout)])
            if i in attn_at:
                blocks.append(Attn(ch))
            self.down.append(blocks); c = ch; skips += [ch, ch]
            if i < len(chans) - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skips.append(ch)

        self.mid1 = ResBlock(c, c, emb, dropout=dropout)
        self.midA = Attn(c)
        self.mid2 = ResBlock(c, c, emb, dropout=dropout)

        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            blocks = nn.ModuleList()
            for _ in range(3):
                blocks.append(ResBlock(c + skips.pop(), ch, emb, dropout=dropout))
                c = ch
            if i in attn_at:
                blocks.append(Attn(ch))
            self.up.append(blocks)
            if i > 0:
                self.upsample.append(nn.Upsample(scale_factor=2, mode='nearest'))

        self.outn = nn.GroupNorm(min(8, c), c)
        self.outc = nn.Conv2d(c, out_ch, 3, padding=1)
        nn.init.zeros_(self.outc.weight); nn.init.zeros_(self.outc.bias)

    def forward(self, x, c_noise):
        e = self.temb(c_noise)
        h = self.inp(x); hs = [h]
        for i, blocks in enumerate(self.down):
            for b in blocks:
                h = b(h, e) if isinstance(b, ResBlock) else b(h)
                if isinstance(b, ResBlock):
                    hs.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h); hs.append(h)
        h = self.mid2(self.midA(self.mid1(h, e)), e)
        for j, blocks in enumerate(self.up):
            for b in blocks:
                if isinstance(b, ResBlock):
                    h = b(torch.cat([h, hs.pop()], dim=1), e)
                else:
                    h = b(h)
            if j < len(self.upsample):
                h = self.upsample[j](h)
        return self.outc(Fn.silu(self.outn(h)))

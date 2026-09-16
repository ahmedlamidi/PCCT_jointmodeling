"""Generator and discriminator for the Morovati et al. 2025 baseline.

doi 10.1088/1361-6560/adaf71, "Patch-based dual-domain photon-counting CT data
correction with residual-based WGAN-ViT".

Generator   fully convolutional: residual blocks, skip connections, transposed
            convolutions for upsampling. NOT a ViT -- the ViT appears only as the
            perceptual-loss feature extractor. Deterministic: one output per input.
            "Residual-based": "the final output produced by adding the processed
            data back to the input", i.e. out = in + f(in).

Discriminator  4 residual blocks, 64 -> 128 -> 256 -> 512 filters, then dense.

Input and output are 16x16x9 patches of energy-bin counts, normalised by
diffusion/norm_stats.py so the two arms are directly comparable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, norm=True):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.InstanceNorm2d(cout, affine=True) if norm else nn.Identity()
        self.n2 = nn.InstanceNorm2d(cout, affine=True) if norm else nn.Identity()
        self.skip = (nn.Conv2d(cin, cout, 1, stride=stride)
                     if (cin != cout or stride != 1) else nn.Identity())

    def forward(self, x):
        h = Fn.leaky_relu(self.n1(self.c1(x)), 0.2)
        h = self.n2(self.c2(h))
        return Fn.leaky_relu(h + self.skip(x), 0.2)


class Generator(nn.Module):
    """out = in + f(in). Encoder / bottleneck / decoder with skip connections."""

    def __init__(self, ch=9, base=64, n_res=6):
        super().__init__()
        self.inp = nn.Conv2d(ch, base, 3, padding=1)
        self.d1 = ResBlock(base, base * 2, stride=2)         # 16 -> 8
        self.d2 = ResBlock(base * 2, base * 4, stride=2)     # 8  -> 4
        self.body = nn.Sequential(*[ResBlock(base * 4, base * 4) for _ in range(n_res)])
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.r2 = ResBlock(base * 4, base * 2)               # cat with d1 output
        self.u1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.r1 = ResBlock(base * 2, base)                   # cat with inp output
        self.out = nn.Conv2d(base, ch, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h0 = Fn.leaky_relu(self.inp(x), 0.2)
        h1 = self.d1(h0)
        h2 = self.d2(h1)
        b = self.body(h2)
        u = self.r2(torch.cat([self.u2(b), h1], 1))
        u = self.r1(torch.cat([self.u1(u), h0], 1))
        return x + self.out(u)                                # residual


class Discriminator(nn.Module):
    """4 residual blocks 64->128->256->512, then dense. No sigmoid (WGAN critic).

    NO normalisation in the critic (every block is built with norm=False; an earlier
    version of this docstring wrongly said InstanceNorm). BatchNorm is ruled out: the
    gradient penalty is applied per sample, and BatchNorm makes the critic's output
    depend on the rest of the batch, which invalidates it (Gulrajani et al. 2017,
    section 4). The GENERATOR's ResBlocks do use InstanceNorm. Neither choice is
    stated in Morovati et al.
    """

    def __init__(self, ch=9, base=64, patch=16):
        super().__init__()
        self.body = nn.Sequential(
            ResBlock(ch, base, stride=1, norm=False),
            ResBlock(base, base * 2, stride=2, norm=False),      # 16 -> 8
            ResBlock(base * 2, base * 4, stride=2, norm=False),  # 8  -> 4
            ResBlock(base * 4, base * 8, stride=2, norm=False),  # 4  -> 2
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(base * 8 * (patch // 8) ** 2, 256),
                                  nn.LeakyReLU(0.2), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.body(x))


class SmallViT(nn.Module):
    """Feature extractor for the perceptual loss.

    Morovati train the ViT from scratch on PCCT data rather than using an
    ImageNet-pretrained network: "Pre-trained CNNs are typically trained on
    ImageNet dataset, which may differ significantly from our PCCT dataset."
    So there is no checkpoint to download -- this must be trained (or the
    perceptual term disabled with --lam_perc 0).

    16x16 patches with 4x4 tokens -> 16 tokens.
    """

    def __init__(self, ch=9, dim=192, depth=4, heads=3, patch=4, img=16):
        super().__init__()
        self.embed = nn.Conv2d(ch, dim, patch, stride=patch)
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, depth)

    def forward(self, x):
        t = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        return self.enc(t)

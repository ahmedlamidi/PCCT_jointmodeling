"""Generator and critic losses for the Morovati et al. 2025 baseline.

Generator loss (their Eq 2), four terms:

    L_G = adversarial + lam_mse * MSE + lam_rmae * RMAE + lam_perc * perceptual

    adversarial  -E[ D(G(m)) ]          WGAN critic score on the generated patch
    MSE          E|G(m) - p|^2
    RMAE         E| (G(m) - p) / (p + eps) |     relative mean absolute error
    perceptual   E|| V(G(m)) - V(p) ||^2         V = ViT features, trained on PCCT data

Reported weights: reconstruction terms ~1000, perceptual 10, eps = 1e-4. The paper's
term-to-symbol assignment is ambiguous in the text we could access, so the weights
are exposed as flags rather than hard-coded -- see README.

Critic loss is WGAN with GRADIENT PENALTY (Gulrajani et al. 2017,
arXiv:1704.00028), which the paper states explicitly ("acts as a gradient penalty
for network regularization ... enforces a Lipschitz constraint"), not the weight
clipping of the original WGAN (Arjovsky et al. 2017, arXiv:1701.07875).
"""
import torch


def gradient_penalty(critic, real, fake):
    """E[(||grad D(x_hat)|| - 1)^2] on points interpolated between real and fake."""
    b = real.shape[0]
    eps = torch.rand(b, 1, 1, 1, device=real.device, dtype=real.dtype)
    x = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d = critic(x)
    g = torch.autograd.grad(d.sum(), x, create_graph=True)[0]
    return ((g.reshape(b, -1).norm(dim=1) - 1.0) ** 2).mean()


def critic_loss(critic, real, fake, lam_gp=10.0):
    """WGAN-GP critic objective. Minimised."""
    return critic(fake).mean() - critic(real).mean() + lam_gp * gradient_penalty(
        critic, real, fake.detach() if fake.requires_grad else fake)


def generator_loss(critic, gen_out, target, lam_adv=1.0, lam_mse=1000.0,
                   lam_rmae=1000.0, lam_perc=10.0, eps=1e-4, vit=None,
                   to_phys=None, rmae_floor=1.0):
    """Returns (total, dict of the individual terms) for logging.

    RMAE is a RELATIVE error, so it is only meaningful on a positive physical
    quantity. The network works in standardised log1p space, where values cross
    zero -- dividing by |p| + 1e-4 there concentrates the loss on near-zero pixels:
    measured on real patches, the top 1% of pixels carried 34% of the total RMAE
    weight, and RMAE outweighed MSE ~5x in the total loss. So `to_phys` maps back
    to counts and the denominator is floored at `rmae_floor` counts. This deviates
    from the paper's eps = 1e-4, which was applied to positive projection data.
    """
    adv = -critic(gen_out).mean()
    mse = ((gen_out - target) ** 2).mean()
    if to_phys is not None:
        g, p = to_phys(gen_out), to_phys(target)
        rmae = ((g - p).abs() / (p.abs() + rmae_floor)).mean()
    else:
        rmae = ((gen_out - target).abs() / (target.abs() + eps)).mean()
    perc = torch.zeros((), device=gen_out.device, dtype=gen_out.dtype)
    if vit is not None and lam_perc > 0:
        perc = ((vit(gen_out) - vit(target)) ** 2).mean()
    total = lam_adv * adv + lam_mse * mse + lam_rmae * rmae + lam_perc * perc
    f = lambda t: t.detach().item()            # float() on a grad tensor warns every step
    return total, dict(adv=f(adv), mse=f(mse), rmae=f(rmae), perc=f(perc), total=f(total))

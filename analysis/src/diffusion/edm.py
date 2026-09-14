"""EDM: preconditioning, loss and sampler.

Karras, Aittala, Aila, Laine, "Elucidating the Design Space of Diffusion-Based
Generative Models", NeurIPS 2022.  arXiv:2206.00364

Table 1, EDM column:
    c_skip(s)  = sd^2 / (s^2 + sd^2)
    c_out(s)   = s * sd / sqrt(s^2 + sd^2)
    c_in(s)    = 1 / sqrt(s^2 + sd^2)
    c_noise(s) = ln(s) / 4
    D(x;s)     = c_skip*x + c_out * F(c_in*x, c_noise)
    weight(s)  = (s^2 + sd^2) / (s*sd)^2
    training   ln(s) ~ N(P_mean, P_std^2)

With this preconditioning the weighted loss collapses to a plain MSE on the raw
network output F against  (y - c_skip*x_noisy) / c_out.  That identity is what
makes EDM stable, so the loss below is written in exactly that form.

CONDITIONAL variant: the condition is concatenated as extra input channels and
is NOT scaled by c_in -- c_in preconditions the noisy latent only.
"""
import numpy as np
import torch


class EDMPrecond(torch.nn.Module):
    """Wraps a raw network F to give the denoiser D(x; sigma, cond)."""

    def __init__(self, model, sigma_data=1.0):
        super().__init__()
        self.model = model
        self.sigma_data = float(sigma_data)

    def forward(self, x, sigma, cond=None):
        sigma = sigma.reshape(-1, 1, 1, 1).to(x.dtype)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sd ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log().flatten() / 4.0
        inp = c_in * x
        if cond is not None:
            inp = torch.cat([inp, cond], dim=1)
        F = self.model(inp, c_noise)
        return c_skip * x + c_out * F

    # ---- the same split, exposed for the loss ---------------------------
    def coeffs(self, sigma):
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sd ** 2 + sigma ** 2).sqrt()
        return c_skip, c_out, c_in

    def raw(self, x, sigma, cond=None):
        sigma = sigma.reshape(-1, 1, 1, 1).to(x.dtype)
        _, _, c_in = self.coeffs(sigma)
        inp = c_in * x
        if cond is not None:
            inp = torch.cat([inp, cond], dim=1)
        return self.model(inp, sigma.log().flatten() / 4.0)


def edm_loss(net, y, cond=None, P_mean=-1.2, P_std=1.2):
    """EDM training loss. y is the CLEAN target, already normalised."""
    rnd = torch.randn(y.shape[0], 1, 1, 1, device=y.device)
    sigma = (rnd * P_std + P_mean).exp()
    n = torch.randn_like(y) * sigma
    x = y + n
    c_skip, c_out, _ = net.coeffs(sigma)
    target = (y - c_skip * x) / c_out            # what F must predict
    F = net.raw(x, sigma.flatten(), cond)
    return ((F - target) ** 2).mean()


# ------------------------------------------------------------------ sampler
def sigma_schedule(n, sigma_min=0.002, sigma_max=80.0, rho=7.0, device='cpu'):
    """EDM Eq 5 time steps, decreasing, with a trailing zero."""
    i = torch.arange(n, dtype=torch.float64, device=device)
    a, b = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    s = (a + i / max(n - 1, 1) * (b - a)) ** rho
    return torch.cat([s, torch.zeros(1, dtype=torch.float64, device=device)])


@torch.no_grad()
def edm_sample(net, shape, cond=None, steps=18, sigma_min=0.002, sigma_max=80.0,
               rho=7.0, S_churn=0.0, S_min=0.0, S_max=float('inf'), S_noise=1.0,
               device='cpu', generator=None, dtype=torch.float32):
    """Algorithm 2: 2nd-order Heun. S_churn=0 gives the deterministic sampler.

    Posterior diversity comes from the initial latent, so S_churn=0 is fine and
    is what we use for coverage -- it keeps the map from latent to sample
    deterministic, which makes repeat runs reproducible.
    """
    ts = sigma_schedule(steps, sigma_min, sigma_max, rho, device)
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator) * float(ts[0])
    for i in range(steps):
        t_cur, t_nxt = float(ts[i]), float(ts[i + 1])
        x_hat, t_hat = x, t_cur
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / steps, np.sqrt(2) - 1)
            t_hat = t_cur * (1 + gamma)
            eps = torch.randn(x.shape, device=device, dtype=dtype, generator=generator)
            x_hat = x + (t_hat ** 2 - t_cur ** 2) ** 0.5 * S_noise * eps
        sv = torch.full((shape[0],), t_hat, device=device, dtype=dtype)
        d = (x_hat - net(x_hat, sv, cond)) / t_hat
        x = x_hat + (t_nxt - t_hat) * d
        if t_nxt > 0:                                    # Heun correction
            sv2 = torch.full((shape[0],), t_nxt, device=device, dtype=dtype)
            d2 = (x - net(x, sv2, cond)) / t_nxt
            x = x_hat + (t_nxt - t_hat) * 0.5 * (d + d2)
    return x

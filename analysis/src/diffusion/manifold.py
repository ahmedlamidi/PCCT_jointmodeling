"""The 2-D manifold that ideal 9-bin count vectors live on.

Every noise-free bin count is a function of two numbers -- cm of soft tissue and
cm of bone -- so the 9-vectors do not fill 9-D space. Measured on 1854 rays, an
SVD of the log-counts puts 99.97% of the variance in 2 components:

    component 1  91.58%
    component 2   8.39%   (cumulative 99.97%)
    component 3   0.03%

Consequences for a generative model over bin counts:

  * A sample can be perfectly calibrated in every bin MARGINALLY and still be
    off the manifold, i.e. a count vector no attenuation path can produce.
    Per-bin coverage cannot detect this.
  * A diagonal-Gaussian head is structurally incapable of representing it.
    This is the concrete reason to use a generative model here.

    mf = Manifold.fit(clean_labels)        # (N, 9) noise-free counts
    r  = mf.residual(samples)              # distance off the surface
    z  = mf.coords(samples)                # (N, 2) position on it
"""
import json, os
import numpy as np


class Manifold:
    def __init__(self, mean, basis, sv, dim):
        self.mean, self.basis, self.sv, self.dim = mean, basis, sv, dim

    # ------------------------------------------------------------------
    @staticmethod
    def fit(counts, dim=2, floor=1e-6):
        """counts: (N, Nl) NOISE-FREE bin counts. Fitted in log space."""
        X = np.log(np.maximum(np.asarray(counts, float), floor))
        mu = X.mean(0)
        U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
        return Manifold(mu, Vt[:dim], s, dim)

    def explained(self):
        v = self.sv ** 2 / (self.sv ** 2).sum()
        return np.cumsum(v)

    # ------------------------------------------------------------------
    def coords(self, counts, floor=1e-6):
        """-> (..., dim) position along the manifold."""
        X = np.log(np.maximum(np.asarray(counts, float), floor))
        return (X - self.mean) @ self.basis.T

    def project(self, counts, floor=1e-6):
        """Nearest point ON the manifold, back in count units."""
        X = np.log(np.maximum(np.asarray(counts, float), floor))
        R = (X - self.mean) @ self.basis.T @ self.basis + self.mean
        return np.exp(R)

    def residual(self, counts, floor=1e-6):
        """Distance OFF the manifold, in log-count units. 0 = physically consistent."""
        X = np.log(np.maximum(np.asarray(counts, float), floor))
        D = X - self.mean
        return np.linalg.norm(D - D @ self.basis.T @ self.basis, axis=-1)

    # ------------------------------------------------------------------
    def save(self, path):
        np.savez(path, mean=self.mean, basis=self.basis, sv=self.sv, dim=self.dim)

    @staticmethod
    def load(path):
        d = np.load(path)
        return Manifold(d['mean'], d['basis'], d['sv'], int(d['dim']))


def joint_coverage(S, Y, levels):
    """Joint credible-region coverage for a whole vector, not bin by bin.

    S: (nsamp, N, d) posterior samples.  Y: (N, d) truth.

    Uses the Mahalanobis distance of the truth inside the sample cloud, compared
    against the empirical quantile of the samples' own Mahalanobis distances --
    so it needs no Gaussian assumption, only that the region be nested.
    A model can pass per-bin coverage in every bin and still fail this.

    FINITE-SAMPLE BIAS. With few samples the empirical quantile is optimistic, so
    a PERFECTLY calibrated model still reads low. Measured on synthetic Gaussian
    data (nominal -> empirical):

        nsamp   0.50   0.80   0.90   0.95   0.99
           32  0.477  0.743  0.835  0.891  0.939
           64  0.500  0.776  0.866  0.921  0.967
          256  0.497  0.782  0.886  0.939  0.985
          512  0.496  0.795  0.896  0.951  0.987

    So do NOT read an absolute calibration claim off nsamp=64 -- at the 0.90 level
    it is ~3.4 points low before the model does anything wrong. Comparing two arms
    at the SAME nsamp is fine, since the bias largely cancels.
    """
    S = np.asarray(S, float); Y = np.asarray(Y, float)
    mu = S.mean(0)
    d = S - mu
    cov = np.einsum('sni,snj->nij', d, d) / max(S.shape[0] - 1, 1)
    cov += 1e-9 * np.eye(S.shape[-1])
    inv = np.linalg.inv(cov)
    md_s = np.einsum('sni,nij,snj->sn', d, inv, d)
    dy = Y - mu
    md_y = np.einsum('ni,nij,nj->n', dy, inv, dy)
    return np.array([(md_y <= np.quantile(md_s, L, axis=0)).mean() for L in levels])

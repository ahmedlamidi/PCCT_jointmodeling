"""Fan-beam forward projector and FBP, matching PcTK's geometry.

Geometry (from PcTK 1_inputdata/v_N0.mat):
    R    = 600 mm    source to isocentre
    Rd   = 1080 mm   source to detector
    Nch  = 1854      channels, 225 um pitch, equiangular
    views over a full 360 deg

Forward projection is done in parallel geometry (rotate and sum, which is fast
and exact up to interpolation) and then resampled onto fan coordinates:

    fan ray (beta, gamma)  ->  parallel ray (theta = beta + gamma, t = R sin gamma)

fbp() is the inverse: rebin fan -> parallel, ramp filter, backproject. It is the
same code that produced viewE_recon.png, moved here so there is one copy.

Units: image values are material FRACTIONS (dimensionless). Line integrals come
out in CENTIMETRES, matching PcTK's m4_sino_v.

    from projector import Geometry
    g = Geometry(nview=720)
    sino = g.forward(img_fraction)      # (Nch, nview), cm
    rec  = g.fbp(sino)                  # (npix, npix)
"""
import numpy as np
from scipy.ndimage import rotate, map_coordinates


class Geometry:
    def __init__(self, nview=2000, nch=1854, dpix_mm=0.225, R=600.0, Rd=1080.0,
                 npix=512, fov_mm=250.0):
        self.nview, self.nch, self.npix, self.fov = nview, nch, npix, fov_mm
        self.R, self.Rd = R, Rd
        self.pix_mm = fov_mm / npix
        self.gam = (np.arange(nch) - (nch - 1) / 2) * (dpix_mm / Rd)   # fan angle
        self.beta = np.arange(nview) * 2 * np.pi / nview               # source angle
        self.t = R * np.sin(self.gam)                                  # fan -> parallel offset
        self.tu = np.linspace(self.t[0], self.t[-1], nch)              # uniform t grid

    # ------------------------------------------------------------------
    def forward_parallel(self, img):
        """(npix,npix) fractions -> (nch, nview) parallel line integrals in cm."""
        n = self.npix
        out = np.empty((self.nch, self.nview))
        # column coordinate of the uniform t grid, in pixels
        col = self.tu / self.pix_mm + (n - 1) / 2.0
        for k, th in enumerate(self.beta):
            r = rotate(img, np.degrees(th), reshape=False, order=1, mode='constant', cval=0.0)
            line = r.sum(axis=0) * self.pix_mm / 10.0                  # mm -> cm
            out[:, k] = np.interp(col, np.arange(n), line, left=0.0, right=0.0)
        return out

    def par_to_fan(self, p):
        """Resample a parallel sinogram (nch, nview) onto fan coordinates."""
        dth = self.beta[1] - self.beta[0]
        dt = self.tu[1] - self.tu[0]
        TH = (self.beta[None, :] + self.gam[:, None]) % (2 * np.pi)
        T = np.repeat(self.t[:, None], self.nview, 1)
        ci = (TH / dth) % self.nview                                    # wraps in view
        ri = (T - self.tu[0]) / dt
        return map_coordinates(p, [ri.ravel(), ci.ravel()], order=1,
                               mode='grid-wrap').reshape(self.nch, self.nview)

    def forward(self, img):
        """(npix,npix) fractions -> (nch, nview) FAN line integrals in cm."""
        return self.par_to_fan(self.forward_parallel(img))

    # ------------------------------------------------------------------
    def fbp(self, g, window=None):
        """Fan sinogram (nch, nview) -> image. Rebin, ramp filter, backproject.
        window: see fbp_parallel."""
        out = np.empty_like(g)
        for i in range(self.nch):                                       # fan -> parallel
            out[i] = np.interp((self.beta - self.gam[i]) % (2 * np.pi),
                               self.beta, g[i], period=2 * np.pi)
        p = np.stack([np.interp(self.tu, self.t, out[:, k]) for k in range(self.nview)], 1)
        return self.fbp_parallel(p, window)

    def fbp_parallel(self, p, window=None):
        """Parallel sinogram (nch, nview) on the uniform t grid self.tu -- what
        forward_parallel() produces -- -> image. Ramp filter, backproject.

        window=None: plain ramp up to the DETECTOR Nyquist (the original behaviour).
        window='hann': ramp x Hann, cut off at the IMAGE Nyquist 1/(2*pix_mm) (Kak &
        Slaney, Principles of Computerized Tomographic Imaging, ch. 3). The channel
        pitch at isocentre (0.125 mm) is ~8x finer than a 256-pixel image (0.98 mm),
        so with the plain ramp, noise the image cannot represent aliases into it: a
        noisy count sinogram reconstructs as noise (measured, see ssim_eval.py)."""
        dt = self.tu[1] - self.tu[0]
        npad = 1 << int(np.ceil(np.log2(2 * self.nch)))
        f = np.fft.fftfreq(npad)                                         # cycles/sample
        filt = 2 * np.abs(f) / dt
        if window == 'hann':
            fc = dt / (2 * self.pix_mm)                                  # image Nyquist, cycles/sample
            filt = filt * np.where(np.abs(f) < fc, 0.5 * (1 + np.cos(np.pi * f / fc)), 0.0)
        elif window is not None:
            raise ValueError('window must be None or "hann", got %r' % (window,))
        pf = np.real(np.fft.ifft(np.fft.fft(p, npad, axis=0) * filt[:, None], axis=0))[:self.nch]

        ax = (np.arange(self.npix) - (self.npix - 1) / 2) * self.pix_mm
        X, Y = np.meshgrid(ax, ax)
        img = np.zeros((self.npix, self.npix))
        for k in range(self.nview):
            tp = X * np.cos(self.beta[k]) + Y * np.sin(self.beta[k])
            img += np.interp(tp, self.tu, pf[:, k], left=0, right=0)
        return img * (np.pi / self.nview) * 10.0                        # cm^-1 -> fraction/cm scale

    # ------------------------------------------------------------------
    def roundtrip_check(self, img):
        """Project then reconstruct. Geometry is right if this returns the phantom."""
        rec = self.fbp(self.forward(img))
        m = img > 0.05
        s = rec[m].mean() / max(img[m].mean(), 1e-9)                    # global scale
        err = np.abs(rec / max(s, 1e-9) - img)
        return dict(scale=s, mean_abs_err=float(err[m].mean()),
                    rel_err=float(err[m].mean() / max(img[m].mean(), 1e-9)), rec=rec)

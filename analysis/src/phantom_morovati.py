"""Random material phantoms after Morovati et al. 2025 (doi 10.1088/1361-6560/adaf71).

Their description (Methods, read via IOPscience):
    "3D Shepp-Logan phantoms with random shapes and multiple material compositions"
    "five ellipsoids of different materials fully enclosed within a sphere"
    materials "soft tissue, adipose tissue, brain gray matter, white matter, blood,
    and cortical bone", water filling the gaps, air outside
    "256 x 256 x 256 cube with a voxel size of 0.113 mm^3"

What they do NOT state, and what is chosen here (each is an argument):
    sphere radius        0.9 of the cube half-width (26 mm across), leaving an air
                         margin -- pile-up is largest in air
    ellipsoid semi-axes  uniform in 0.15-0.45 of the sphere radius
    orientation          uniformly random 3D rotation (QR of a Gaussian matrix,
                         Mezzadri 2007, arXiv math-ph/0609050)
    placement            uniform, subject to "fully enclosed": the ellipsoid's
                         bounding sphere lies inside the sphere
    "different materials" read as 5 of the 6 drawn without replacement
    overlaps             the later ellipsoid wins, so each voxel is ONE material.
                         Classic Shepp-Logan values add, but the sum of two
                         tissues' attenuations is not a tissue.
    grey vs white matter both ICRU-44 brain (NIST has one entry) -- identical here

Returns per-voxel brain- and bone-equivalent fractions (a, b from
nist_materials.two_basis), so projecting them gives the two path lengths
Forward takes.
"""
import numpy as np
import nist_materials as NM

MATERIALS = ['soft_tissue', 'adipose', 'grey_matter', 'white_matter', 'blood', 'cortical_bone']
NIST_NAME = dict(soft_tissue='soft_tissue', adipose='adipose', grey_matter='brain',
                 white_matter='brain', blood='blood', cortical_bone='bone', water='water')
LABELS = ['air', 'water'] + MATERIALS        # label value -> material name


def _rotation(rng):
    """Uniform random rotation in 3D (Haar measure on SO(3))."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def random_phantom(n=256, rng=None, sphere_r=0.9, nell=5, axis_lo=0.15, axis_hi=0.45, slab=32):
    """-> labels (n, n, n) uint8 indexing LABELS, and the ellipsoid list (for the manifest).

    Coordinates are voxel centres in [-1, 1] across the cube, axis order (z, y, x).
    Built in z-slabs to keep peak memory at ~100 MB for n = 256.
    """
    rng = np.random.default_rng() if rng is None else rng
    ax = ((np.arange(n) - (n - 1) / 2) / (n / 2)).astype(np.float32)
    Y2, X2 = np.meshgrid(ax, ax, indexing='ij')
    lab = np.zeros((n, n, n), np.uint8)
    for z0 in range(0, n, slab):
        z = ax[z0:z0 + slab, None, None]
        lab[z0:z0 + slab][(z * z + Y2 * Y2 + X2 * X2) <= sphere_r ** 2] = 1      # water
    ells = []
    for m in rng.choice(len(MATERIALS), size=nell, replace=False):
        semi = rng.uniform(axis_lo, axis_hi, 3) * sphere_r
        room = sphere_r - semi.max() - 2.0 / n              # one voxel inside the wall
        d = rng.standard_normal(3)
        c = d / np.linalg.norm(d) * room * rng.uniform() ** (1 / 3)   # uniform in a ball
        R = _rotation(rng)
        ells.append(dict(material=MATERIALS[m], centre_zyx=c.round(4).tolist(),
                         semi_axes=semi.round(4).tolist(), rotation=R.round(5).tolist()))
        for z0 in range(0, n, slab):
            zz = ax[z0:z0 + slab]
            P = np.stack(np.broadcast_arrays(zz[:, None, None] - c[0], Y2[None] - c[1],
                                             X2[None] - c[2]), -1)
            inside = (((P @ R) / semi) ** 2).sum(-1) <= 1.0     # ellipsoid's own frame
            lab[z0:z0 + slab][inside] = 2 + m
    return lab, ells


def equivalent_maps(lab, coef=None):
    """labels -> (brain-equivalent, bone-equivalent) fraction per voxel, float32.

    One cm of material m counts as a_m cm of NIST brain plus b_m cm of NIST
    cortical bone. Air is (0, 0).
    """
    coef = NM.two_basis() if coef is None else coef
    a = np.zeros(len(LABELS), np.float32)
    b = np.zeros_like(a)
    for i, name in enumerate(LABELS[1:], 1):
        a[i], b[i] = coef[NIST_NAME[name]][:2]
    return a[lab], b[lab]

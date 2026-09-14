"""Two-material Shepp-Logan style head phantom.

Morovati trained on 3D Shepp-Logan variants; this is the 2D two-material
equivalent, sized so a central ray sees roughly 20 cm of soft tissue and 2 cm of
bone -- the same operating point as PcTK's shipped m4_sino_v, and the point at
which the CRLB sweep and the ablation tables were computed.

Returns FRACTION maps (0-1), not densities. projector.Geometry.forward turns
them into line integrals in cm.

    from phantom import head, variants
    brain, bone = head(512)
    for brain, bone in variants(512, n=10, seed=0): ...
"""
import numpy as np

# (x0, y0, a, b, angle_deg, material, value)   normalised coordinates, |x|,|y| <= 1
BASE = [
    (0.00,  0.000, 0.850, 0.920,   0, 'bone',  1.00),   # skull, outer
    (0.00,  0.000, 0.790, 0.870,   0, 'bone', -1.00),   # skull, inner -> makes a ring
    (0.00,  0.000, 0.790, 0.870,   0, 'brain', 1.00),   # brain
    (0.00,  0.350, 0.210, 0.250,   0, 'brain', -0.25),  # ventricle
    (0.00, -0.100, 0.410, 0.160,   0, 'brain', -0.15),
    (-0.22, 0.000, 0.110, 0.310, -18, 'brain',  0.15),
    (0.22,  0.000, 0.160, 0.410,  18, 'brain',  0.15),
    (0.00, -0.605, 0.046, 0.023,   0, 'bone',   0.60),  # calcification
    (-0.08,-0.605, 0.046, 0.023,   0, 'bone',   0.60),
    (0.06, -0.605, 0.023, 0.046,   0, 'bone',   0.60),
]


def _draw(spec, n):
    ax = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(ax, -ax)
    out = {'brain': np.zeros((n, n)), 'bone': np.zeros((n, n))}
    for x0, y0, a, b, ang, mat, val in spec:
        th = np.radians(ang)
        xr = (X - x0) * np.cos(th) + (Y - y0) * np.sin(th)
        yr = -(X - x0) * np.sin(th) + (Y - y0) * np.cos(th)
        out[mat] += val * ((xr / a) ** 2 + (yr / b) ** 2 <= 1.0)
    # Shepp-Logan ellipse values ADD where they overlap. Bone must saturate at
    # 1.0 (pure cortical bone) or an insert landing on the skull ring stacks on
    # top of it and produces rays through 13-16 cm of bone -- more than a skull
    # is wide, and outside the pile-up grid's range.
    return np.clip(out['brain'], 0, None), np.clip(out['bone'], 0, 1.0)


def head(n=512):
    """Nominal phantom. -> (brain_fraction, bone_fraction), each (n, n)."""
    return _draw(BASE, n)


def _spec_at_z(spec, zf):
    """Slice an ellipsoid stack at normalised height zf in [-1, 1].

    An ellipsoid with semi-axes (a, b, c) cut at height z is an ellipse with
    semi-axes a*s, b*s where s = sqrt(1 - (z/c)^2). Using c = 1 for every
    structure keeps the head roughly spherical in z, which is enough for a
    Shepp-Logan style phantom.
    """
    s = np.sqrt(max(1.0 - zf * zf, 0.0))
    if s <= 0:
        return []
    return [(x0 * s, y0 * s, a * s, b * s, ang, mat, val)
            for x0, y0, a, b, ang, mat, val in spec]


def head3d(n=256, nz=64, spec=None):
    """3D phantom as a stack of axial slices.

    -> brain (nz, n, n), bone (nz, n, n).  Slice k is at normalised height
    zf = -1 + 2*(k+0.5)/nz, so the stack is a solid of revolution in z.
    """
    spec = BASE if spec is None else spec
    br = np.zeros((nz, n, n), np.float32)
    bo = np.zeros((nz, n, n), np.float32)
    for k in range(nz):
        zf = -1.0 + 2.0 * (k + 0.5) / nz
        sl = _spec_at_z(spec, zf * 0.92)          # 0.92 keeps the poles non-degenerate
        if not sl:
            continue
        a, b = _draw(sl, n)
        br[k], bo[k] = a, b
    return br, bo


def variants3d(n=256, nz=64, count=10, seed=0):
    """Randomised 3D variants, using the same coupled-skull jitter as variants()."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        spec = []
        cx, cy = rng.normal(0, 0.03), rng.normal(0, 0.03)
        ao = BASE[0][2] * rng.uniform(0.93, 1.07)
        bo_ = BASE[0][3] * rng.uniform(0.93, 1.07)
        wall = rng.uniform(0.040, 0.075)
        ai, bi = ao - wall, bo_ - wall
        spec.append((cx, cy, ao, bo_, 0.0, 'bone', 1.00))
        spec.append((cx, cy, ai, bi, 0.0, 'bone', -1.00))
        spec.append((cx, cy, ai, bi, 0.0, 'brain', 1.00))
        for x0, y0, a, b, ang, mat, val in BASE[3:]:
            spec.append((x0 + cx + rng.normal(0, 0.03), y0 + cy + rng.normal(0, 0.03),
                         a * rng.uniform(0.93, 1.07), b * rng.uniform(0.93, 1.07),
                         ang + rng.normal(0, 6), mat, val))
        for _ in range(rng.integers(0, 5)):
            r = rng.uniform(0.02, 0.06)
            ang = rng.uniform(0, 2 * np.pi); rad = rng.uniform(0.1, 0.50)
            spec.append((rad * np.cos(ang) + cx, rad * np.sin(ang) + cy, r,
                         r * rng.uniform(0.6, 1.6), rng.uniform(0, 180), 'bone',
                         rng.uniform(0.3, 0.9)))
        yield head3d(n, nz, spec)


def variants(n=512, count=10, seed=0):
    """Randomised Shepp-Logan variants for training data.

    Jitters ellipse centres, radii and angles, and adds 0-4 extra bone inserts.
    """
    rng = np.random.default_rng(seed)
    for _ in range(count):
        spec = []
        # The skull is a RING: entries 0 (outer bone), 1 (inner bone, negative)
        # and 2 (brain) must stay coupled. Jittering them independently lets the
        # outer grow while the inner shrinks, giving a wall several times too
        # thick -- a tangential ray then sees 13-16 cm of bone, more than a
        # skull is wide. Draw the outer ellipse and a WALL THICKNESS instead.
        cx, cy = rng.normal(0, 0.03), rng.normal(0, 0.03)
        ao = BASE[0][2] * rng.uniform(0.93, 1.07)
        bo = BASE[0][3] * rng.uniform(0.93, 1.07)
        wall = rng.uniform(0.040, 0.075)            # ~5-9 mm at a 125 mm half-FOV
        ai, bi = ao - wall, bo - wall
        spec.append((cx, cy, ao, bo, 0.0, 'bone', 1.00))
        spec.append((cx, cy, ai, bi, 0.0, 'bone', -1.00))
        spec.append((cx, cy, ai, bi, 0.0, 'brain', 1.00))
        for x0, y0, a, b, ang, mat, val in BASE[3:]:
            spec.append((x0 + cx + rng.normal(0, 0.03), y0 + cy + rng.normal(0, 0.03),
                         a * rng.uniform(0.93, 1.07), b * rng.uniform(0.93, 1.07),
                         ang + rng.normal(0, 6), mat, val))
        for _ in range(rng.integers(0, 5)):                 # extra calcifications
            r = rng.uniform(0.02, 0.06)
            # radius capped well inside the skull ring (inner semi-axes 0.79/0.87)
            # so inserts sit in soft tissue rather than stacking on the ring
            ang = rng.uniform(0, 2 * np.pi); rad = rng.uniform(0.1, 0.50)
            spec.append((rad * np.cos(ang), rad * np.sin(ang), r,
                         r * rng.uniform(0.6, 1.6), rng.uniform(0, 180), 'bone',
                         rng.uniform(0.3, 0.9)))
        yield _draw(spec, n)

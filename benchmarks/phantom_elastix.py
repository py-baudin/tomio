""" synthetic image pair with a known smooth deformation, for tomio.elastix benchmarks

Geometries of the dixon (moving: 224x224x64, 1x1x5 mm) and T1 (fixed: 256x288x40, 0.98x0.98x9.6 mm) images.
Tissue classes from thresholded random blobs inside an elliptic body, rendered with two contrasts
(non-monotonic intensity mapping between them), texture (~2 mm) in the moving image only, noise.
The fixed image is the deformed scene: fixed(x) = contrast_F(x + u(x)), moving(y) = contrast_M(y),
so the true transform (fixed -> moving) is T(x) = x + u(x).

usage:
    from phantom_elastix import make_pair
    fixed, moving, u, body = make_pair()  # u: true displacement at the fixed grid points (npoint x 3), body: mask
"""
import pathlib

import numpy as np

from tomio import elastix

FIXED = dict(shape=(256, 288, 40), spacing=(0.98, 0.98, 9.6), origin=(-125.0, -140.0, -190.0))
MOVING = dict(shape=(224, 224, 64), spacing=(1.0, 1.0, 5.0), origin=(-112.0, -112.0, -160.0))
LEVELS_F = [1.0, 0.6, 0.3, 0.8]  # fixed contrast of the 4 tissue classes
LEVELS_M = [0.2, 0.9, 0.5, 0.1]  # moving contrast (non-monotonic w.r.t. the fixed one)
CACHE = pathlib.Path(__file__).parent / "data"


def blobs(points, n, seed, wmin, wmax, chunk=2**16):
    """sum of n random gaussian blobs with random signs"""
    rng = np.random.default_rng(seed)
    centers = rng.uniform([-100, -100, -170], [100, 100, 170], size=(n, 3))
    widths = rng.uniform(wmin, wmax, size=(n, 1))
    weights = rng.uniform(-1, 1, size=n)
    out = np.zeros(len(points))
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk]
        out[start : start + chunk] = (weights * np.exp(-0.5 * (((p[:, None] - centers) / widths) ** 2).sum(-1))).sum(1)
    return out


def tissues(points):
    """body mask and 4 tissue class memberships (soft)"""
    body = 1 / (1 + np.exp(((points[:, 0] / 95) ** 2 + (points[:, 1] / 80) ** 2 - 1) * 20))
    sigmoid = lambda x: 1 / (1 + np.exp(-x / 0.02))
    a, b = sigmoid(blobs(points, 160, 1, 8, 20)), sigmoid(blobs(points, 160, 2, 6, 18))
    return body, [body * a * b, body * a * (1 - b), body * (1 - a) * b, body * (1 - a) * (1 - b)]


def render(points, levels, texture=0.0, noise=0.03, seed=0, grid=None):
    """image values at points: tissue contrast, optional texture (~2 mm), noise"""
    body, classes = tissues(points)
    values = sum(level * cls for level, cls in zip(levels, classes))
    rng = np.random.default_rng(seed)
    if texture:
        shape, spacing = grid
        tex = elastix.gaussian_filter(rng.standard_normal(shape), [2.0 / s for s in spacing]).ravel()
        values = values + texture / tex.std() * body * tex
    return values + noise * rng.standard_normal(len(points)), body


def deformation(points, amplitude=6.0, seed=3, chunk=2**16):
    """smooth displacement (sum of 12 wide gaussian bumps), max |u_i| = amplitude (mm)"""
    rng = np.random.default_rng(seed)
    centers = rng.uniform([-80, -80, -150], [80, 80, 150], size=(12, 3))
    widths = rng.uniform(30, 50, size=(12, 1))
    directions = rng.normal(size=(12, 3))
    u = np.zeros(points.shape)
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk]
        u[start : start + chunk] = np.exp(-0.5 * (((p[:, None] - centers) / widths) ** 2).sum(-1)) @ directions
    return u * amplitude / np.abs(u).max()


def make_pair(texture=0.05, same_contrast=False, amplitude=6.0, cache=True):
    """fixed and moving Images, true displacement u at the fixed points, body mask of the fixed image"""
    name = f"pair_t{texture}_s{int(same_contrast)}_a{amplitude}.npz"
    path = CACHE / name
    if cache and path.exists():
        data = np.load(path)
        fval, mval, u, body = data["fixed"], data["moving"], data["u"], data["body"]
    else:
        fpoints = elastix.Geometry(**FIXED).points
        mpoints = elastix.Geometry(**MOVING).points
        u = deformation(fpoints, amplitude=amplitude)
        fval, fbody = render(fpoints + u, LEVELS_F, seed=10)
        if same_contrast:
            mval, _ = render(mpoints, LEVELS_F, seed=12)
        else:
            mval, _ = render(mpoints, LEVELS_M, texture=texture, seed=11, grid=(MOVING["shape"], MOVING["spacing"]))
        body = fbody > 0.5
        if cache:
            CACHE.mkdir(exist_ok=True)
            np.savez_compressed(path, fixed=fval, moving=mval, u=u, body=body)
    fixed = elastix.Image(fval.reshape(FIXED["shape"]), spacing=FIXED["spacing"], origin=FIXED["origin"])
    moving = elastix.Image(mval.reshape(MOVING["shape"]), spacing=MOVING["spacing"], origin=MOVING["origin"])
    return fixed, moving, u, body

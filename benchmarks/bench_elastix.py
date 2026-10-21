""" benchmark tomio.elastix: rigid registration of synthetic volumes with the dixon/T1 geometries

usage: python benchmarks/bench_elastix.py [metric ...]   (default: mse mi)

fixed: 256x288x40 at 0.98x0.98x9.6 mm (T1), moving: 224x224x64 at 1x1x5 mm (dixon), same smooth phantom,
moving shifted and rotated (known transform). Prints the best time of 3 runs, the number of cost function
evaluations per pyramid level and the error of the recovered transform (mm, at the fixed grid points).
"""
import sys
import time

import numpy as np

from tomio import elastix

FIXED = dict(shape=(256, 288, 40), spacing=(0.98, 0.98, 9.6), origin=(-125.0, -140.0, -190.0))
MOVING = dict(shape=(224, 224, 64), spacing=(1.0, 1.0, 5.0), origin=(-112.0, -112.0, -160.0))
TRUE = np.array([0.0, 0.0, 0.05, 2.0, -3.0, 4.0])  # Euler angles (rad), translation (mm)


def phantom(points, nblob=40, seed=0):
    """smooth phantom: sum of gaussian blobs in world coordinates"""
    rng = np.random.default_rng(seed)
    centers = rng.uniform([-90, -90, -140], [90, 90, 140], size=(nblob, 3))
    widths = rng.uniform(10, 30, size=(nblob, 1))
    weights = rng.uniform(0.5, 1, size=nblob)
    values = np.zeros(len(points))
    for chunk in range(0, len(points), 2**18):
        sub = points[chunk : chunk + 2**18]
        dist2 = (((sub[:, np.newaxis] - centers) / widths) ** 2).sum(axis=-1)
        values[chunk : chunk + 2**18] = (weights * np.exp(-0.5 * dist2)).sum(axis=1)
    return values


def make_images():
    fixed_geometry = elastix.Geometry(**FIXED)
    moving_geometry = elastix.Geometry(**MOVING)
    trans = elastix.Transformation("rigid", fixed_geometry, TRUE)
    # moving(T(x)) = fixed(x): sample the phantom at T^-1(y)
    inverse = elastix.Transformation("rigid", fixed_geometry, elastix.TransformRigid().inverse(TRUE))
    fixed = phantom(fixed_geometry.points).reshape(FIXED["shape"])
    moving = phantom(inverse(moving_geometry.points)).reshape(MOVING["shape"])
    return elastix.Image(fixed, **{k: FIXED[k] for k in ("spacing", "origin")}), \
        elastix.Image(moving, **{k: MOVING[k] for k in ("spacing", "origin")}), trans


def main(metrics):
    fixed, moving, true = make_images()
    points = fixed.geometry.points[:: 97]
    for metric in metrics:
        best, result = np.inf, None
        for _ in range(3):
            t0 = time.perf_counter()
            result = elastix.Elastix(elastix.Config(transform="rigid", metric=metric)).execute(fixed, moving)
            best = min(best, time.perf_counter() - t0)
        trans, info = result
        error = np.linalg.norm(trans(points) - true(points), axis=1)
        nfev = ", ".join(f"{res.level}: {res.nfev}" for res in info.levels)
        print(f"{metric:4s} {best:6.2f} s | evaluations per level {nfev} | error mean {error.mean():.3f} mm, max {error.max():.3f} mm")


if __name__ == "__main__":
    main(sys.argv[1:] or ["mse", "mi"])

""" benchmark interpolation backends: scipy vs SimpleITK (oblique target, float32)

usage: python benchmarks/bench_interp.py [size ...]

For each method, prints the time of each backend, and the differences between
their results: in the interior (sources coordinates >= 2 voxels from the border)
and in the whole volume.
Memory is not reported: SimpleITK allocations are not seen by tracemalloc.
"""
import sys
import time

import numpy as np
import SimpleITK as sitk

from iomed import volume, interp

METHODS = ["nearest", "linear", "spline3"]


def timed(func, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        best = min(best, time.perf_counter() - t0)
    return best, result


def interior_mask(target, source, margin=2):
    """target voxels whose position is >= `margin` voxels inside the source"""
    indices = np.indices(target.shape).reshape(3, -1).T
    coords = source.get_indices(target.get_coords(indices))
    inside = np.all((coords >= margin) & (coords <= np.array(source.shape) - 1 - margin), axis=1)
    return inside.reshape(target.shape)


def main(sizes):
    angle = np.radians(10)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    print(f"SimpleITK {sitk.Version_VersionString()}, threads: {sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()}")
    print(f"{'size':>6} {'method':>8} {'scipy':>8} {'sitk':>8} {'speedup':>8} {'max diff (interior)':>20} {'max diff (all)':>15} {'voxels differing':>17}")
    for n in sizes:
        rng = np.random.default_rng(0)
        src = volume.Volume(rng.random((n, n, n), dtype=np.float32))
        tgt = volume.Volume(np.zeros((n, n, n), dtype=np.float32), origin=(0.3 * n, -0.1 * n, 0.5), transform=rotation.T)
        interior = interior_mask(tgt, src)
        for method in METHODS:
            t_scipy, res_scipy = timed(lambda: interp.interpolate_like(tgt, src, method=method, backend="scipy"))
            t_sitk, res_sitk = timed(lambda: interp.interpolate_like(tgt, src, method=method, backend="sitk"))
            diff = np.abs(res_scipy.A.astype(float) - res_sitk.A)
            print(
                f"{n:>5}³ {method:>8} {t_scipy:>7.2f}s {t_sitk:>7.2f}s {t_scipy / t_sitk:>7.1f}x "
                f"{diff[interior].max():>20.2e} {diff.max():>15.2e} {np.mean(diff > 1e-5):>16.2%}"
            )


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [64, 128, 192])

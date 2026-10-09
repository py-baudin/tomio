""" benchmark interpolation (oblique target, spline3, float32)

usage: python benchmarks/bench_interp.py [size ...]
"""
import sys
import time
import tracemalloc

import numpy as np

from iomed import volume, interp


def measure(func):
    tracemalloc.start()
    t0 = time.perf_counter()
    func()
    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return duration, peak


def main(sizes):
    angle = np.radians(10)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    print(f"{'size':>6} {'time':>9} {'peak memory':>12} {'bytes/voxel':>12}")
    for n in sizes:
        src = volume.Volume(np.random.rand(n, n, n).astype(np.float32))
        tgt = volume.Volume(np.zeros((n, n, n), dtype=np.float32), transform=rotation.T)
        duration, peak = measure(lambda: interp.interpolate_like(tgt, src, method="spline3"))
        print(f"{n:>5}³ {duration:>7.2f} s {peak / 2**20:>9.0f} MB {peak / n**3:>12.1f}")


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [64, 128, 192])

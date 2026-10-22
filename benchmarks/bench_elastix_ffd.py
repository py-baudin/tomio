""" benchmark tomio.elastix free-form (B-spline) registration on a synthetic pair with a known deformation

usage: python benchmarks/bench_elastix_ffd.py [case ...]   (default: all)

cases:
    a: cross-contrast + texture, MI
    b: same contrast, no texture, MSE
    c: cross-contrast + texture, NGF
For each configuration: error vs the true displacement at body points (mean, per axis, mm), held-out metric
(new random samples) of the result and of the true deformation, runtime, number of parameters.
"""
import sys
import time

import numpy as np

from tomio import elastix
from phantom_elastix import make_pair

CASES = {
    "a": dict(pair=dict(texture=0.05, same_contrast=False), metric="mi"),
    "b": dict(pair=dict(texture=0.0, same_contrast=True), metric="mse"),
    "c": dict(pair=dict(texture=0.05, same_contrast=False), metric="ngf"),
}

CONFIGS = {
    "lbfgs 3000": dict(optimizer="lbfgs", sampler={"maxsample": 3000}, pyramid={"downsample": True}),
    "sgd 2048": dict(optimizer="sgd", sampler={"maxsample": 2048}, pyramid={"downsample": True}),
    "defaults": dict(),  # sgd, smoothing pyramid (B-spline defaults)
}


def heldout(fixed, moving, transform, parameters, metric, nsample=100_000, seed=1):
    """metric with new random samples (full resolution)"""
    config = elastix.Config(metric=metric, sampler={"maxsample": nsample})
    sampler = config.setup("sampler")
    sampler._rgen = np.random.default_rng(seed)
    objective = elastix.Objective(fixed, moving, transform, sampler, config.setup("metric"), config.setup("interpolator"))
    return objective(parameters)


def true_parameters(transform, u_func):
    """B-spline coefficients interpolating the true displacement at the control points"""
    shape = tuple(transform.grid["shape"])
    disp = u_func(transform.control_points())
    return np.concatenate([elastix.bspline_coefficients(disp[:, a].reshape(shape), 3).ravel() for a in range(3)])


def main(cases, grid_spacing=20.0):
    from phantom_elastix import deformation

    for case in cases:
        settings = CASES[case]
        fixed, moving, u, body = make_pair(**settings["pair"])
        points = fixed.geometry.points[body.ravel()][::50]
        upoints = u[body.ravel()][::50]
        print(f"case {case} ({settings['metric']}): identity error {np.linalg.norm(upoints, axis=1).mean():.2f} mm")
        reference = elastix.TransformBSpline(grid_spacing=grid_spacing)
        reference.setup(fixed.geometry)
        ptrue = true_parameters(reference, deformation)
        hout_true = heldout(fixed, moving, reference, ptrue, settings["metric"])
        hout_id = heldout(fixed, moving, reference, np.zeros_like(ptrue), settings["metric"])
        print(f"   held-out metric: identity {hout_id:.4f}, true deformation {hout_true:.4f}")
        for label, options in CONFIGS.items():
            config = {"transform": {"type": "bspline", "grid_spacing": grid_spacing}, "metric": settings["metric"], **options}
            t0 = time.perf_counter()
            trans, info = elastix.Elastix(elastix.Config(config)).execute(fixed, moving)
            dt = time.perf_counter() - t0
            error = trans(points) - points - upoints
            hout = heldout(fixed, moving, trans.transform, trans.parameters, settings["metric"])
            print(
                f"   {label:12s} {dt:6.1f} s, {trans.parameters.size} param | error mean {np.linalg.norm(error, axis=1).mean():.2f} mm"
                f" (x {np.abs(error[:, 0]).mean():.2f}, y {np.abs(error[:, 1]).mean():.2f}, z {np.abs(error[:, 2]).mean():.2f})"
                f" | held-out {hout:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES))

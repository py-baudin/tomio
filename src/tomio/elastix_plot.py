""" plots of registration results (tomio.elastix), requires matplotlib

usage:
    from tomio import elastix, elastix_plot
    registered, trans = elastix.register(fixed, moving, transform="rigid", metric="mi", return_transform=True)
    fig = elastix_plot.plot_registration(fixed, moving, trans)
    fig.savefig("registration.png")
"""
import numpy as np

from . import elastix


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires matplotlib (pip install matplotlib)") from None
    return plt


def _image(obj):
    return obj if isinstance(obj, elastix.Image) else elastix.Image(obj)


def _normalize(array, low=1, high=99):
    """display range from percentiles"""
    array = np.abs(array) if np.iscomplexobj(array) else np.asarray(array, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return np.zeros(array.shape)
    vmin, vmax = np.percentile(finite, [low, high])
    return np.clip((array - vmin) / (vmax - vmin + 1e-12), 0, 1)


def _slices(shape, axis, slices):
    """slice indices along `axis`"""
    if np.ndim(slices) == 0:
        n = int(slices)
        size = shape[axis]
        return sorted({int(round(f * (size - 1))) for f in np.linspace(0.25, 0.75, n)}) if n > 1 else [size // 2]
    return list(slices)


def _take(array, axis, index):
    """2d slice (displayed with the first remaining axis horizontal)"""
    if array.ndim == 2:
        return array.T
    return np.take(array, index, axis=axis).T


def _aspect(spacing, axis, ndim):
    axes = [d for d in range(ndim) if d != axis] if ndim == 3 else [0, 1]
    return spacing[axes[1]] / spacing[axes[0]]


def _overlay(fixed, moving):
    """fixed in magenta, moving in green"""
    return np.stack([fixed, moving, fixed], axis=-1)


def _checkerboard(fixed, moving, tiles=8):
    rows, cols = fixed.shape
    r = (np.arange(rows) * tiles // max(rows, 1))[:, None]
    c = (np.arange(cols) * tiles // max(cols, 1))[None, :]
    return np.where((r + c) % 2 == 0, fixed, moving)


def _identity(geometry):
    return elastix.Transformation("translation", geometry, np.zeros(geometry.ndim))


def _global_part(transformation):
    """composition of the global stages of a transformation (None if there is none)"""
    if isinstance(transformation, elastix.TransformationSequence):
        stages = [trans for trans in transformation if trans.is_global]
        return elastix.TransformationSequence(stages) if stages else None
    return transformation if transformation.is_global else None


def local_displacement(transformation, points):
    """displacement of the non-global stages (B-spline) at points: T(x) - T_global(x)"""
    transformation = elastix.as_transformation(transformation)
    moved = transformation(points)
    reference = _global_part(transformation)
    return moved - (points if reference is None else reference(points))


def plot_registration(fixed, moving, transformation, *, slices=3, axis=2, method="linear", grid=True, title=None, tiles=8):
    """figure of a registration result, one row per slice of the fixed image

    columns: fixed, moving before registration (resampled on the fixed grid), registered, overlays before and
    after (fixed: magenta, moving: green), checkerboard (after); for non-global transforms: local deformation
    magnitude (mm) with the deformed grid

    fixed, moving: images (tomio Volume, elastix Image or array)
    transformation: Transformation, TransformationSequence or serialized transformation (fixed -> moving)
    slices: number of slices (evenly spaced between 25% and 75%) or list of indices along `axis`
    axis: slice axis of the fixed image (3d)
    method: interpolation of the moving image (see `elastix.transform`)
    grid: draw the deformed grid on the deformation column
    """
    plt = _pyplot()
    fixed, moving = _image(fixed), _image(moving)
    transformation = elastix.as_transformation(transformation)
    geometry = fixed.geometry
    before = np.asarray(elastix.transform(_identity(geometry), moving, ref=fixed, method=method))
    after = np.asarray(elastix.transform(transformation, moving, ref=fixed, method=method))
    deformable = not transformation.is_global

    ndim = fixed.ndim
    indices = _slices(fixed.shape, axis, slices) if ndim == 3 else [0]
    aspect = _aspect(fixed.spacing, axis, ndim)
    fixed_n, before_n, after_n = _normalize(fixed.array), _normalize(before), _normalize(after)

    columns = ["fixed", "moving (before)", "registered", "overlay before", "overlay after", "checkerboard"]
    if deformable:
        columns.append("local deformation (mm)")
    fig, axes = plt.subplots(len(indices), len(columns), figsize=(2.6 * len(columns), 2.6 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        f, b, a = (_take(arr, axis, index) for arr in (fixed_n, before_n, after_n))
        panels = [f, b, a, _overlay(f, b), _overlay(f, a), _checkerboard(f, a, tiles)]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1, aspect=aspect, interpolation="nearest")
        if deformable:
            ax = axes[row, len(panels)]
            _plot_deformation(ax, transformation, geometry, axis, index, aspect, grid=grid, background=f)
        label = f"slice {index}" if ndim == 3 else ""
        axes[row, 0].set_ylabel(label)
    for col, name in enumerate(columns):
        axes[0, col].set_title(name, fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def _plot_deformation(ax, transformation, geometry, axis, index, aspect, grid=True, background=None, step=8):
    """local displacement magnitude on a slice, with the deformed grid"""
    shape = tuple(geometry.shape)
    ndim = len(shape)
    if ndim == 3:
        coords = np.stack(np.meshgrid(*[np.arange(n) if d != axis else [index] for d, n in enumerate(shape)], indexing="ij"), -1)
        coords = coords.reshape(-1, 3)
        plane = [d for d in range(3) if d != axis]
        size = [shape[d] for d in plane]
    else:
        coords = np.reshape(np.indices(shape), (2, -1)).T
        plane, size = [0, 1], list(shape)
    points = geometry.to_points(coords)
    disp = local_displacement(transformation, points)
    magnitude = np.linalg.norm(disp, axis=1).reshape(size).T
    if background is not None:
        ax.imshow(background, cmap="gray", vmin=0, vmax=1, aspect=aspect, interpolation="nearest")
    image = ax.imshow(magnitude, cmap="viridis", alpha=0.6 if background is not None else 1.0, aspect=aspect, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    if grid:
        # displacement in pixel units of the displayed axes
        pixels = disp @ np.linalg.inv(geometry.matrix).T
        u = pixels[:, plane[0]].reshape(size)
        v = pixels[:, plane[1]].reshape(size)
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing="ij")
        for i in range(0, size[0], step):
            ax.plot(x[i] + u[i], y[i] + v[i], color="w", lw=0.5)
        for j in range(0, size[1], step):
            ax.plot(x[:, j] + u[:, j], y[:, j] + v[:, j], color="w", lw=0.5)
        ax.set_xlim(-0.5, size[0] - 0.5)
        ax.set_ylim(size[1] - 0.5, -0.5)


def plot_levels(info, *, title=None):
    """cost and number of cost evaluations per pyramid level (`info` returned by `Elastix.execute`)"""
    plt = _pyplot()
    levels = getattr(info, "levels", [])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    labels = [str(res.level) for res in levels]
    ax1.plot(labels, [res.fun for res in levels], "o-")
    ax1.set_xlabel("pyramid level")
    ax1.set_ylabel("cost")
    ax2.bar(labels, [res.nfev or 0 for res in levels])
    ax2.set_xlabel("pyramid level")
    ax2.set_ylabel("cost evaluations")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig

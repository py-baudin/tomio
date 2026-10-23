"""interpolation of sparse ROIs (labels drawn on some slices)"""

import numpy as np
import SimpleITK as sitk

from .api import _output
from .config import Config
from .image import Image
from .pipeline import Elastix
from .splines import ndspline_interpolation
from .transforms import TransformationSequence


# registration of consecutive slices (method "registration" of `interpolate_slices`)
SLICE_REGISTRATION = {
    "transform": {"type": "bspline", "grid_spacing": 20.0, "bending": 1.0},
    "metric": "mi",
    # 2 pyramid levels, 32 iterations: as good as 4 levels and 128 iterations on muscle MRI, 0.3 s per slice pair
    "pyramid": {"maxlevel": 1},
    "optimizer": {"type": "sgd", "niter": 32},
}


def interpolate_slices(roi, image=None, *, method="shape", axis=None, slices=None, config=None):
    """fill the slices of a sparse ROI (label map drawn on some slices) between its labelled slices

    Each label is interpolated as a signed distance map (shape-based interpolation): at a slice between the
    labelled slices lo and hi, the distance maps of lo and hi are blended with weights (hi - k) / (hi - lo) and
    (k - lo) / (hi - lo); each voxel takes the label of the most negative (inside) blended distance. A label drawn
    on one side only vanishes half-way.

    roi: label map (array, tomio Volume or Image), 0: background
    image: image on the ROI grid (method "registration")
    method: "shape": signed distance interpolation;
        "registration": the distance maps of lo and hi are first warped onto slice k along the image, by
        deformable 2D registrations of consecutive slices (k -> k - 1 -> ... -> lo, and k -> k + 1 -> ... -> hi),
        as Ogier et al. (2017) for muscle MRI; slower, better with large gaps
    axis: slice axis (default: as `warp_roi`, the most labelled axis)
    slices: labelled slice indices (default: slices with labels); other slices between them are replaced
    config: registration settings of consecutive slices (default: `SLICE_REGISTRATION`)

    returns: tomio Volume (3d) or Image (2d) on the ROI grid; slices outside the labelled range are unchanged
    """
    roi = Image(roi, label="roi")
    array = np.asarray(roi.array)
    ndim = array.ndim
    mask = array > 0
    if axis is None:
        counts = [mask.sum(axis=tuple(j for j in range(ndim) if j != i)) for i in range(ndim)]
        axis = int(np.argmax([np.max(c) for c in counts]))
    if slices is None:
        slices = np.flatnonzero(np.moveaxis(mask, axis, 0).reshape(array.shape[axis], -1).any(axis=1))
    slices = sorted(int(k) for k in slices)
    if method not in ("shape", "registration"):
        raise ValueError(f"Unknown method: {method}")
    if method == "registration" and image is None:
        raise ValueError("method 'registration' needs an image")

    stack = np.moveaxis(array, axis, -1)  # slices last
    out = np.array(stack)
    labels = [int(v) for v in np.unique(stack[..., slices]) if v > 0]
    inplane = [a for a in range(ndim) if a != axis]
    spacing = np.asarray(roi.spacing, dtype=float)[inplane]

    if method == "registration":
        images = np.moveaxis(np.asarray(Image(image).array, dtype=float), axis, -1)
        if images.shape != stack.shape:
            raise ValueError("The image must have the shape of the ROI")
        pairs = {}

        def pair(fixed, moving):
            """2D transformation: slice `fixed` -> slice `moving`"""
            if (fixed, moving) not in pairs:
                f = Image(images[..., fixed], spacing=spacing)
                m = Image(images[..., moving], spacing=spacing)
                pipeline = Elastix(Config(config if config is not None else SLICE_REGISTRATION))
                pairs[fixed, moving], _ = pipeline.execute(f, m)
            return pairs[fixed, moving]

    def distances(source):
        """signed distance maps of the labels of a labelled slice"""
        return {label: _signed_distance(stack[..., source] == label, spacing) for label in labels}

    for lo, hi in zip(slices[:-1], slices[1:]):
        if hi - lo < 2:
            continue
        dlo, dhi = distances(lo), distances(hi)
        for k in range(lo + 1, hi):
            w = (k - lo) / (hi - lo)
            if method == "registration":
                wlo = _warp_distances(dlo, [pair(j, j - 1) for j in range(k, lo, -1)], spacing)
                whi = _warp_distances(dhi, [pair(j, j + 1) for j in range(k, hi)], spacing)
            else:
                wlo, whi = dlo, dhi
            out[..., k] = _fuse_distances(wlo, whi, w, labels)

    values = np.moveaxis(out, -1, axis)
    return _output(values, roi.geometry, label="roi")


def _signed_distance(mask, spacing):
    """signed distance map of a 1D or 2D mask (mm, negative inside, zero half-way between the boundary pixels),
    None if empty"""
    if not np.any(mask):
        return None
    shape = mask.shape
    spacing = [float(v) for v in spacing]
    if mask.ndim == 1:
        mask, spacing = mask[:, np.newaxis], spacing + [1.0]
    image = sitk.GetImageFromArray(np.ascontiguousarray(mask.T, dtype=np.uint8))
    image.SetSpacing(spacing)
    distance = sitk.SignedMaurerDistanceMap(image, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    # Maurer's map is 0 at the inner boundary pixels: shift by half a pixel
    return sitk.GetArrayFromImage(distance).T.astype(float).reshape(shape) - 0.5 * min(spacing)


def _warp_distances(distances, chain, spacing):
    """distance maps warped through a chain of 2D transformations (points of the target slice -> source slice)"""
    if not chain:
        return distances
    sequence = TransformationSequence(chain)
    points = sequence()
    warped = {}
    for label, distance in distances.items():
        if distance is None:
            warped[label] = None
            continue
        # outside the slice: nearest edge value
        geometry = Image(distance, spacing=spacing).geometry
        values = ndspline_interpolation(distance, geometry.to_coords(points), order=1, mode="nearest")
        warped[label] = values.reshape(distance.shape)
    return warped


def _fuse_distances(dlo, dhi, w, labels):
    """label map from the blended signed distances of two slices"""
    shape = next(d.shape for d in list(dlo.values()) + list(dhi.values()) if d is not None)
    best = np.full(shape, np.inf)
    out = np.zeros(shape, dtype=int)
    for label in labels:
        a, b = dlo[label], dhi[label]
        if a is None and b is None:
            continue
        # label on one side only: the other side is a constant equal to its depth (vanishes half-way)
        if a is None:
            a = np.full(shape, -b.min())
        if b is None:
            b = np.full(shape, -a.min())
        blended = (1 - w) * a + w * b
        better = (blended < 0) & (blended < best)
        out[better], best[better] = label, blended[better]
    return out

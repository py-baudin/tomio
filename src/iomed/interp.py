# coding=utf-8
""" interpolate volumes from metadata """

import numpy as np

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

# local imports
from . import volume

NAX = np.newaxis


class InterpolationError(Exception):
    pass



def interpolate_like(target, source, **kwargs):
    """interpolate source volume to match target volume"""
    target = volume.asvolume(target)
    source = volume.asvolume(source)
    target_geom = {'origin': target.origin, 'spacing': target.spacing, 'transform': target.transform, 'shape': target.shape}
    source_geom = {'origin': source.origin, 'spacing': source.spacing, 'transform': source.transform, 'shape': source.shape}
    return interpolate(target_geom, source_geom, source, **kwargs)



def interpolate_to(geometry, src, **kwargs):
    """interpolate source volume to target geometry
    {'origin': ..., 'spacing': ..., 'transform': ...}
    """
    src = volume.asvolume(src)
    ndim = src.ndim
    tgt_geom = dict(geometry)
    src_geom = {
        "origin": src.origin,
        "spacing": src.spacing,
        "transform": src.transform,
        "shape": src.shape,
    }

    if not "origin" in tgt_geom:
        tgt_geom["origin"] = src_geom["origin"]
    if not "spacing" in tgt_geom:
        tgt_geom["spacing"] = src_geom["spacing"]
    if not "transform" in tgt_geom:
        tgt_geom["transform"] = src_geom["transform"]
    if not "shape" in tgt_geom:
        # bounding box of the source voxel centers, in target indices
        Ts = volume.make_affine_transform(src.origin, src.spacing, src.transform)
        Tt = volume.make_affine_transform(
            tgt_geom["origin"], tgt_geom["spacing"], tgt_geom["transform"]
        )
        idx = np.indices((2,) * ndim).reshape(ndim, -1).T * (np.array(src.shape) - 1)
        loc = np.linalg.solve(
            Tt, Ts @ np.concatenate([idx, np.ones((2**ndim, 1))], axis=1).T
        )[:3]
        # snap the box to the target grid, and move the origin to its first corner
        lower = np.round(np.min(loc, axis=1))
        upper = np.round(np.max(loc, axis=1))
        tgt_geom["shape"] = tuple((upper - lower).astype(int) + 1)
        tgt_geom["origin"] = tuple((Tt @ np.r_[lower, 1])[:3])

    return interpolate(tgt_geom, src_geom, src, **kwargs)


def interpolate_roi(ref, roi, *, disjoint=True, axis="auto", method=None):
    """Interpolate mask/ROI into `ref` geometry

    when downsampling: use nearest nonzero slice
    args:
        disjoint: if upsampling, keep slices disjoint (no interpolation in the slice axis)
        axis: slice axis ("auto": guessed from the ROI)
        method: None: each ROI voxel is copied to the nearest `ref` voxel(s)
            interpolation method (e.g. "spline1", "spline3"): higher quality upsampling
            each label (incl. background) is interpolated separately as a binary mask,
            and each voxel gets the label with the highest interpolated value
    """
    roi = volume.asvolume(roi)
    ref = volume.asvolume(ref)

    ndim = roi.ndim

    # compute transform from roi to ref referentials
    Ts = volume.make_affine_transform(roi.origin, roi.spacing, roi.transform)
    Tt = volume.make_affine_transform(ref.origin, ref.spacing, ref.transform)
    Tref = np.linalg.inv(Tt) @ Ts

    # new ROI
    new = np.zeros_like(ref, dtype=roi.dtype)

    # ROI coordinates in ref referentials
    nonzero = np.asarray(np.nonzero(roi))
    coords = apply_transform(Tref, nonzero)

    # axis
    mask = roi > 0
    if axis == "auto":
        # find most likely slice axis
        axis = np.argmax(
            [
                np.max(np.sum(mask, axis=tuple(j for j in range(ndim) if j != i)))
                for i in range(ndim)
            ]
        )
    else:
        assert axis in list(range(ndim))

    if method is not None:
        return interpolate_labels(ref, roi, method=method, round_axes=[axis] if disjoint else [])

    if np.any(np.array(roi.spacing) > np.array(ref.spacing)):
        # fix upsampling 'gaps'
        pixcoords = apply_transform(Tref, np.ones(3)) - apply_transform(
            Tref, np.zeros(3)
        )
        if disjoint:
            pixcoords[axis] = 1
        pix = np.meshgrid(
            *tuple(np.arange(c) for i, c in enumerate(pixcoords)), indexing="ij"
        )
        pix = np.reshape(pix, (ndim, -1))
        nonzero = np.stack([np.repeat(nonzero[d], pix[0].size) for d in range(ndim)])
        coords = np.stack(
            [np.ravel(coords[d, :, NAX] + pix[d, NAX]) for d in range(ndim)]
        )

    # round to nearest grid coordinates
    indices = (coords + 0.5).astype(int)

    # remove indices outside volume
    valid = np.all((indices.T >= 0) & (indices.T < new.shape), axis=1)
    nonzero = nonzero[:, valid]
    indices = indices[:, valid]

    # remove duplicate indices
    indices, keep = np.unique(indices, axis=1, return_index=True)
    new[tuple(indices)] = roi[tuple(nonzero[:, keep])]

    return new


def interpolate_labels(ref, labelmap, *, method="spline1", round_axes=()):
    """Interpolate label map into `ref` geometry, label by label

    Each label (incl. background: 0) is interpolated separately as a binary mask;
    each voxel gets the label with the highest interpolated value.
    Outside the label map: background.

    args:
        method: interpolation method (e.g. "spline1", "spline3")
        round_axes: label map axes in which coordinates are rounded (no interpolation)
    """
    if ndimage is None:
        raise ImportError('`scipy` is required for interpolation')
    ref = volume.asvolume(ref)
    labelmap = volume.asvolume(labelmap)
    order = get_order(method)
    resample = make_resampler(get_geometry(ref), get_geometry(labelmap), round_axes=round_axes)

    new = np.zeros(ref.shape, dtype=labelmap.dtype)
    best = np.full(ref.shape, -np.inf, dtype=np.float32)
    for label in np.union1d([0], labelmap):
        mask = (labelmap.A == label).astype(np.float32)
        # outside the label map: background
        cval = 1.0 if label == 0 else 0.0
        value = resample(mask, order=order, mode="constant", cval=cval, prefilter=order > 1)
        update = value > best
        new[update] = label
        best[update] = value[update]

    return volume.Volume(new, spacing=ref.spacing, origin=ref.origin, transform=ref.transform)


def interpolate(
    target_geom,
    source_geom,
    *sources,
    method=None,
    order=None,
    mode=None,
    cval=0,
):
    """Interpolate a series of volume based on the target
    and source tags (see Volume in crisio)

    Tag dictionaries should contain the following fields:
        'origin', 'transform', 'spacing', 'shape'

    Parameters:
    ---
        target_geom: dict
        source_geom: dict
        sources: ndarray (or list of ndarrays)
            The arrays dimensions must match the values in source_geom

        method: [None], 'nearest', 'spline1', 'spline3'...
            Interpolation method
            Default: 'nearest' for integer and bool data, 'spline3' otherwise
        order: spline order (overrides `method`)
        mode: [None], 'nearest', any float number
            How values outside the box defined by the coordinates
            should be filled. Default: constant 0

    Returns
    ---
        results: Volume (or list of)
            Interpolated array, with the same dtype as the source

    """
    if ndimage is None:
        raise ImportError('`scipy` is required for interpolation')

    # extend mode
    try:
        const_value = float(mode)
        mode = "constant"
    except (TypeError, ValueError):
        # mode is not a number
        const_value = cval

    if mode is None:
        mode = "constant"
    elif not mode in ["constant", "nearest", "mirror", "wrap"]:
        raise ValueError('Invalid value for `mode`: "%s"' % mode)

    resample = make_resampler(target_geom, source_geom)

    # do the interpolation
    results = []
    for vol in sources:
        vol = np.asarray(vol)
        dtype = vol.dtype
        # interpolation method
        vol_order = get_order(method or get_default_method(vol), order)

        if dtype == bool:
            # not supported by scipy.ndimage
            vol = vol.astype(np.uint8)
        isnan = np.isnan(vol) if np.issubdtype(dtype, np.inexact) else None
        if isnan is not None and np.any(isnan):
            vol = np.where(isnan, 0, vol)  # copy: do not modify the input
        else:
            isnan = None

        res = resample(vol, order=vol_order, mode=mode, cval=const_value, prefilter=vol_order > 1)
        if isnan is not None:
            mask = resample(isnan.astype(np.uint8), order=0, mode="nearest", prefilter=False)
            res[mask > 0] = np.nan

        results.append(volume.Volume(res.astype(dtype, copy=False), **get_geometry(target_geom, shape=False)))

    # return
    if len(sources) == 1:
        return results[0]
    else:
        return results


# utilities

def get_default_method(array):
    """default interpolation method: 'nearest' for integer and bool data (e.g. label maps), else 'spline3'"""
    dtype = np.asarray(array).dtype
    if dtype == bool or np.issubdtype(dtype, np.integer):
        return "nearest"
    return "spline3"


def get_order(method, order=None):
    """spline order of interpolation method ('nearest', 'spline1', 'spline3'...)"""
    if order is not None:
        return order
    elif method == "nearest":
        return 0
    elif method[:-1] == "spline" and method[-1].isdigit():
        return int(method[-1])
    raise ValueError(f"Invalid interpolation method: {method}")


def get_geometry(obj, shape=True):
    """geometry dict (origin, spacing, transform, [shape]) of a Volume or dict, with defaults"""
    if isinstance(obj, dict):
        geom = {
            "origin": obj.get("origin", (0, 0, 0)),
            "spacing": obj.get("spacing", (1, 1, 1)),
            "transform": obj.get("transform", np.eye(3)),
        }
        if shape:
            geom["shape"] = tuple(obj["shape"])
    else:
        geom = {"origin": obj.origin, "spacing": obj.spacing, "transform": obj.transform}
        if shape:
            geom["shape"] = obj.shape
    return geom


def make_resampler(target_geom, source_geom, round_axes=()):
    """return function resample(array, **kwargs) from source to target geometry

    kwargs: passed to scipy.ndimage (order, mode, cval, prefilter)
    round_axes: source axes in which coordinates are rounded (no interpolation);
        always includes single-slice axes
    """
    target_geom = get_geometry(target_geom)
    source_geom = get_geometry(source_geom)
    shape_t = target_geom["shape"]

    # transform from target indices to source indices
    Tt = volume.make_affine_transform(target_geom["origin"], target_geom["spacing"], target_geom["transform"])
    Ts = volume.make_affine_transform(source_geom["origin"], source_geom["spacing"], source_geom["transform"])
    try:
        M = np.linalg.inv(Ts) @ Tt
    except np.linalg.LinAlgError as exc:
        raise InterpolationError(f"Could not inverse transform: {Ts}")

    rounded = np.array(source_geom["shape"][:3]) == 1
    rounded[list(round_axes)] = True
    if np.any(rounded):
        # rounding cannot be expressed as an affine transform: compute all coordinates
        indices = np.indices(shape_t).reshape(3, -1)
        coords = apply_transform(M, indices)
        coords[rounded] = np.round(coords[rounded])

        def resample(array, **kwargs):
            return ndimage.map_coordinates(array, coords, **kwargs).reshape(shape_t)
    else:
        # affine transform: only the output is allocated
        def resample(array, **kwargs):
            return ndimage.affine_transform(
                array, M[:3, :3], offset=M[:3, 3], output_shape=shape_t, **kwargs
            )
    return resample


def apply_transform(T, coords):
    coords = np.asarray(coords)
    coords = np.concatenate([coords, np.ones((1,) + coords.shape[1:])], axis=0)
    return (T @ coords)[:-1]

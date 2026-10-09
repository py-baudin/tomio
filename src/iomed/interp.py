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


def interpolate_roi(ref, roi, *, disjoint=True, axis="auto"):
    """Interpolate mask/ROI into `ref` geometry

    when downsampling: use nearest nonzero slice
    args:
        disjoint: if upsampling, keep slices disjoint (no interpolation in the slice axis)
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


def interpolate(
    target_geom,
    source_geom,
    *sources,
    method="spline3",
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

        method: ['spline3'], 'nearest'
            Interpolation method
        mode: [None], 'nearest', any float number
            How values outside the box defined by the coordinates
            should be filled. Default: constant 0

    Returns
    ---
        results: ndarray (or list of)
            Interpolated array

    """
    if ndimage is None:
        raise ImportError('`scipy` is required for interpolation')

    # interpolation method
    if order is not None:
        pass
    elif method[:-1] == "spline":
        order = int(method[-1])
    elif method == "nearest":
        order = 0

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

    # target geometry
    geomt = target_geom
    shape_t = geomt["shape"]

    offset_t = geomt.get("origin", (0, 0, 0))
    tmatrix_t = geomt.get("transform", (1, 0, 0, 0, 1, 0, 0, 0, 1))
    spacing_t = geomt.get("spacing", (1, 1, 1))

    # affine transform
    Tt = volume.make_affine_transform(offset_t, spacing_t, tmatrix_t)

    # source geometry
    geoms = source_geom
    shape_s = geoms["shape"]
    offset_s = geoms.get("origin", (0, 0, 0))
    tmatrix_s = geoms.get("transform", (1, 0, 0, 0, 1, 0, 0, 0, 1))
    spacing_s = geoms.get("spacing", (1, 1, 1))

    # affine transform
    Ts = volume.make_affine_transform(offset_s, spacing_s, tmatrix_s)

    # calculate coords
    indi, indj, indk = np.indices(shape_t)

    # locations in space of indices
    pos = Tt.dot(
        np.r_[[indi.ravel()], [indj.ravel()], [indk.ravel()], [np.ones(indi.size)]]
    )

    # coordinates in pixels
    try:
        invTs = np.linalg.inv(Ts)
    except np.linalg.LinAlgError as exc:
        raise InterpolationError(f"Could not inverse transform: {Ts}")
    coords = invTs.dot(pos)[:3, :]

    # single slice: round coordinates
    is_single = np.array(shape_s) == 1
    coords[is_single] = np.round(coords[is_single])

    # do the interpolation
    results = []
    for vol in sources:
        isnan = np.isnan(vol)
        vol = np.where(isnan, 0, vol)  # copy: do not modify the input
        res = ndimage.map_coordinates(
            vol, coords, order=order, mode=mode, cval=const_value, prefilter=True
        ).reshape(shape_t)

        if np.any(isnan):
            mask = ndimage.map_coordinates(
                isnan, coords, order=0, mode="nearest", prefilter=False
            )
            res[mask.reshape(shape_t) > 0.5] = np.nan

        results.append(volume.Volume(res, spacing=geomt['spacing'], origin=geomt['origin'], transform=geomt['transform']))

    # return
    if len(sources) == 1:
        return results[0]
    else:
        return results



def apply_transform(T, coords):
    coords = np.asarray(coords)
    coords = np.concatenate([coords, np.ones((1,) + coords.shape[1:])], axis=0)
    return (T @ coords)[:-1]

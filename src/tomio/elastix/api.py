"""registration and resampling API"""

import numpy as np
import SimpleITK as sitk

from .. import volume as _volume
from .. import interp as _interp
from .config import make_stages
from .image import Image, direction_to_transform
from .interpolators import InterpolatorBSpline
from .pipeline import Elastix
from .splines import ndspline_interpolation
from .transforms import Transformation, TransformationSequence, as_transformation


def register(
    fixed,
    moving,
    config=None,
    *,
    return_transform=False,
    reset=False,
    loglevel="WARN",
    callback=None,
    fixed_mask=None,
    moving_mask=None,
    **sections,
):
    """register moving image(s) onto a fixed image

    fixed: fixed image (array, tomio Volume or Image)
    moving: moving image, or list of moving images
    config: registration settings (see `Config`): Config, dict, or YAML/JSON file;
        a list of settings: successive stages (e.g. rigid, then B-spline), each stage starting from the previous ones
    sections: settings of a single stage, e.g. `transform="rigid"`, `metric={"type": "mi", "nbin": 32}`
    return_transform: also return the transformation(s)
    reset: with several moving images, start each registration from the initial parameters
        (default: start from the parameters found for the previous image)
    loglevel: logging level
    callback: function called after each pyramid level: callback(fixed, moving, parameters, config, level=level)
    fixed_mask: samples are drawn in this mask of the fixed image
    moving_mask: samples falling outside this mask of the moving image(s) are discarded

    returns: registered image(s) (tomio Volume in 3d, Image in 2d), [transformation(s)]
        the transformation is a `Transformation`, or a `TransformationSequence` with several stages
    """
    single = not isinstance(moving, (list, tuple))
    moving = [moving] if single else list(moving)
    stages = make_stages(config, **sections)

    fixed = Image(fixed, mask=fixed_mask, label="fixed")
    moving = [Image(im, mask=moving_mask, label="moving") for im in moving]

    registered = []
    transformations = []
    inits = [None] * len(stages)
    for im in moving:
        found = []
        for stage, init in zip(stages, inits):
            pipeline = Elastix(stage, loglevel=loglevel)
            initial = TransformationSequence(found) if found else None
            trans, info = pipeline.execute(fixed, im, init=init, callback=callback, initial=initial)
            found.append(trans)

        transformation = found[0] if len(found) == 1 else TransformationSequence(found)
        method = stages[-1]["resample"].get("method")
        registered.append(_apply(transformation, im, method=method, label="registered"))
        transformations.append(transformation)

        if not reset:
            # use optimized parameters for next registration
            inits = [trans.parameters for trans in found]

    if single:
        registered = registered[0]
        transformations = transformations[0]

    if return_transform:
        return registered, transformations
    return registered


def transform(transformations, moving, *, ref=None, method=None):
    """apply transformation(s) to image(s)

    transformations: Transformation, TransformationSequence or serialized transformation
        (or a list of them, one per moving image)
    moving: image or list of images
    ref: output grid (default: the fixed image of the transformation)
    method: interpolation method (as `tomio.resample`):
        None: 'nearest' for integer and bool data, 'spline3' otherwise
        'nearest', 'linear', 'spline<n>'
        'roi', 'labels': ROI / label map interpolation (global transforms of 3d images: `tomio.resample`,
            otherwise `warp_roi` / `warp_labels`)

    returns: transformed image(s) (tomio Volume in 3d, Image in 2d)
    """
    single = not isinstance(moving, (list, tuple))
    moving = [moving] if single else list(moving)

    if isinstance(transformations, (Transformation, TransformationSequence, dict)):
        transformations = [transformations] * len(moving)
    elif isinstance(transformations, (list, tuple)) and transformations and isinstance(transformations[0], dict) and "transform" in transformations[0]:
        # serialized sequence
        transformations = [transformations] * len(moving)
    elif single:
        raise ValueError("A single transformation must be given")
    elif len(transformations) != len(moving):
        raise ValueError("Different numbers of moving images and transformations")

    geometry = None if ref is None else Image(ref, label="reference").geometry
    transformed = []
    for trans, im in zip(transformations, moving):
        new = _apply(as_transformation(trans), Image(im, label="moving"), geometry=geometry, method=method, label="transformed")
        transformed.append(new)

    if single:
        transformed = transformed[0]
    return transformed


def _apply(transformation, moving, *, geometry=None, method=None, label=None):
    """resample a moving Image through a transformation

    geometry: output grid (default: transformation's geometry)
    method: see `transform`

    returns: tomio Volume (3d) or Image (2d), on the output grid

    Global transforms of real 3d images are resampled with `tomio.resample`: the output grid mapped by the
    transformation is a (possibly oblique) regular grid in the moving image space.
    """
    moving = Image(moving)
    if geometry is not None:
        transformation = transformation.with_geometry(geometry)
    geometry = transformation.geometry
    if method is None:
        method = _interp.get_default_method(moving.array)

    if transformation.is_global and moving.ndim == 3 and not np.iscomplexobj(moving.array):
        # mapped grid: T(origin + M @ i) = (A @ origin + t) + (A @ M) @ i
        A, t = transformation.affine()
        matrix = A @ geometry.matrix
        spacing = np.linalg.norm(matrix, axis=0)
        mapped = {
            "shape": tuple(int(n) for n in geometry.shape),
            "spacing": spacing,
            "origin": A @ geometry.origin + t,
            "transform": direction_to_transform(matrix / spacing),
        }
        vol = _volume.Volume(moving.array, spacing=moving.spacing, origin=moving.origin, transform=moving.transform)
        values = np.asarray(_interp.resample(vol, geometry=mapped, method=method))
    elif method == "roi":
        values = warp_roi(moving, transformation(), geometry).reshape(tuple(geometry.shape))
    elif method in SITK_INTERPOLATORS and moving.ndim >= 2 and not np.iscomplexobj(moving.array):
        values = _resample_sitk(moving, transformation(), geometry, SITK_INTERPOLATORS[method])
    elif method == "labels":
        values = warp_labels(moving, transformation()).reshape(tuple(geometry.shape))
    else:
        order = _interp.get_order(method)
        interpolator = InterpolatorBSpline(order=order, mode="constant", cval=0)
        values = interpolator(moving, transformation()).reshape(tuple(geometry.shape))
        if not np.issubdtype(moving.array.dtype, np.inexact):
            values = np.rint(values).astype(moving.array.dtype)

    return _output(values, geometry, label=label)


# interpolation methods resampled by SimpleITK (multithreaded) through a displacement field
SITK_INTERPOLATORS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "spline1": sitk.sitkLinear,
    "spline3": sitk.sitkBSpline,
    "labels": sitk.sitkLabelLinear,  # as `warp_labels`: linear interpolation of each label's mask
}


def _resample_sitk(moving, points, geometry, interpolator):
    """values of a moving Image at the mapped points of an output grid (`sitk.Resample` with the displacement
    field points - grid points, exact at the grid points); 0 outside the moving image"""
    shape = tuple(int(n) for n in geometry.shape)
    ndim = len(shape)

    def as_sitk(array, geo, vector=False):
        """sitk image of an (x, y, z) array (with a last vector axis)"""
        axes = tuple(range(ndim))[::-1] + ((ndim,) if vector else ())
        image = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(array, axes)), isVector=vector)
        image.SetOrigin([float(v) for v in geo.origin])
        image.SetSpacing([float(v) for v in geo.spacing])
        image.SetDirection([float(v) for v in np.ravel(geo.direction)])
        return image

    field = as_sitk((points - geometry.points).reshape(shape + (ndim,)).astype(np.float64), geometry, vector=True)
    reference = sitk.Image(field)  # output grid (copy: the transform takes the field's buffer)
    transform = sitk.DisplacementFieldTransform(field)

    array = np.asarray(moving.array)
    dtype = array.dtype
    source = as_sitk(array.astype(np.uint8) if dtype == bool else array, moving.geometry)
    result = sitk.Resample(source, reference, transform, interpolator, 0.0, source.GetPixelID())
    values = np.transpose(sitk.GetArrayFromImage(result), tuple(range(ndim))[::-1])
    return values.astype(dtype)


def _output(values, geometry, label=None):
    """array on a grid -> tomio Volume (3d) or Image (2d)"""
    if geometry.ndim == 3:
        return _volume.Volume(values, spacing=geometry.spacing, origin=geometry.origin, transform=direction_to_transform(geometry.direction))
    return Image(values, spacing=geometry.spacing, origin=geometry.origin, direction=geometry.direction, label=label)


def _output_geometry(transformation, ref=None):
    transformation = as_transformation(transformation)
    if ref is not None:
        transformation = transformation.with_geometry(Image(ref, label="reference").geometry)
    return transformation, transformation.geometry


def warp_labels(labelmap, points):
    """label map at points: linear interpolation of each label's mask, label with the largest value"""
    array = np.asarray(labelmap.array)
    coords = labelmap.geometry.to_coords(points)
    best = np.full(len(points), -np.inf)
    out = np.zeros(len(points), dtype=array.dtype)
    for label in np.unique(array):
        mask = (array == label).astype(float)
        values = ndspline_interpolation(mask, coords, order=1, mode="constant", cval=float(label == 0))
        better = values > best
        out[better], best[better] = label, values[better]
    return out


def warp_roi(roi, points, geometry):
    """ROI at the points of an output grid, for ROIs drawn on some slices only (as `tomio.resample(method="roi")`)

    Whole-slice logic: each labelled slice of the ROI (along its slice axis: the most labelled axis) goes to a
    single output slice (along the output axis closest to the ROI slice axis): the one whose points map closest to
    it on average, if within its extent. All the points of that output slice take the labels of that ROI slice
    (nearest neighbor in-plane); output slices without a labelled slice are empty.

    points: the output grid points, in the order of `geometry.points`, mapped into the ROI space
    """
    array = np.asarray(roi.array)
    shape = tuple(int(n) for n in geometry.shape)
    if len(points) != int(np.prod(shape)):
        raise ValueError("warp_roi: the points must be the (transformed) points of the output grid")
    coords = roi.geometry.to_coords(points)
    ndim = array.ndim
    mask = array > 0
    if not np.any(mask):
        return np.zeros(len(points), dtype=array.dtype)
    # slice axis and labelled slices (as interp.interpolate_roi)
    counts = [mask.sum(axis=tuple(j for j in range(ndim) if j != i)) for i in range(ndim)]
    axis = int(np.argmax([np.max(c) for c in counts]))
    labelled = np.flatnonzero(counts[axis])
    # output slice axis: the closest to the ROI slice axis
    alignment = np.abs(geometry.direction.T @ roi.geometry.direction[:, axis])
    out_axis = int(np.argmax(alignment))
    slice_index = np.reshape(np.indices(shape)[out_axis], -1)
    nslice = shape[out_axis]
    # mean position of each output slice along the ROI slice axis (ROI pixels)
    position = np.bincount(slice_index, weights=coords[:, axis], minlength=nslice) / np.bincount(slice_index, minlength=nslice)
    # half extent of an output slice along the ROI slice axis (ROI pixels)
    matrix = np.linalg.inv(roi.geometry.matrix) @ geometry.matrix
    half = max(0.5 * np.sum(np.abs(matrix[axis])), 0.5)
    # each labelled slice -> one output slice (the closest); each output slice keeps its closest labelled slice
    assigned = np.full(nslice, -1)
    distance = np.full(nslice, np.inf)
    for label_slice in labelled:
        k = int(np.argmin(np.abs(position - label_slice)))
        d = abs(position[k] - label_slice)
        if d <= half + 1e-6 and d < distance[k]:
            assigned[k], distance[k] = label_slice, d
    source = assigned[slice_index]
    snapped = np.array(coords)
    snapped[:, axis] = np.maximum(source, 0)
    values = ndspline_interpolation(array, snapped, order=0, mode="constant", cval=0)
    values[source < 0] = 0
    return values.astype(array.dtype)

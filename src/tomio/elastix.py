""" Image registration: an implementation of the Elastix registration framework

cf:
- https://en.wikipedia.org/wiki/Elastix_(image_registration)
- https://elastix.lumc.nl/doxygen/classelastix_1_1RandomSampler.html
- https://simpleelastix.readthedocs.io/ParameterMaps.html

usage:
    from tomio import elastix
    registered, trans = elastix.register(fixed, moving, transform="rigid", metric="mi", return_transform=True)
    roi_registered = elastix.transform(trans, roi, ref=fixed, method="roi")

configuration: see `Config` and `Config.describe()`
"""

import os
import copy
import math
import inspect
import logging
from abc import ABC, abstractmethod
from types import SimpleNamespace

import yaml
import numpy as np
import SimpleITK as sitk

from . import volume as _volume, interp as _interp, configfile as _configfile

LOGGER = logging.getLogger(__name__)


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
            "transform": (matrix / spacing).T,
        }
        vol = _volume.Volume(moving.array, spacing=moving.spacing, origin=moving.origin, transform=moving.transform)
        values = np.asarray(_interp.resample(vol, geometry=mapped, method=method))
    elif method == "roi":
        values = warp_roi(moving, transformation(), geometry).reshape(tuple(geometry.shape))
    elif method == "labels":
        values = warp_labels(moving, transformation()).reshape(tuple(geometry.shape))
    else:
        order = _interp.get_order(method)
        interpolator = InterpolatorBSpline(order=order, mode="constant", cval=0)
        values = interpolator(moving, transformation()).reshape(tuple(geometry.shape))
        if not np.issubdtype(moving.array.dtype, np.inexact):
            values = np.rint(values).astype(moving.array.dtype)

    if geometry.ndim == 3:
        return _volume.Volume(values, spacing=geometry.spacing, origin=geometry.origin, transform=geometry.direction.T)
    return Image(values, spacing=geometry.spacing, origin=geometry.origin, direction=geometry.direction, label=label)


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

    Along the slice axis (most labelled axis), each point takes the nearest labelled slice if it lies within
    the output voxel's extent; nearest neighbor otherwise.
    """
    array = np.asarray(roi.array)
    coords = roi.geometry.to_coords(points)
    ndim = array.ndim
    mask = array > 0
    if not np.any(mask):
        return np.zeros(len(points), dtype=array.dtype)
    # slice axis and labelled slices (as interp.interpolate_roi)
    counts = [mask.sum(axis=tuple(j for j in range(ndim) if j != i)) for i in range(ndim)]
    axis = int(np.argmax([np.max(c) for c in counts]))
    labelled = np.flatnonzero(counts[axis])
    # nearest labelled slice
    position = np.clip(np.searchsorted(labelled, coords[:, axis]), 1, max(len(labelled) - 1, 1))
    candidates = np.stack([labelled[position - 1], labelled[np.minimum(position, len(labelled) - 1)]], axis=1)
    distance = np.abs(candidates - coords[:, axis, np.newaxis])
    nearest = candidates[np.arange(len(points)), np.argmin(distance, axis=1)]
    # half extent of an output voxel along the slice axis (in ROI pixels)
    matrix = np.linalg.inv(roi.geometry.matrix) @ geometry.matrix
    half = max(0.5 * np.sum(np.abs(matrix[axis])), 0.5)
    snapped = np.array(coords)
    snapped[:, axis] = nearest
    values = ndspline_interpolation(array, snapped, order=0, mode="constant", cval=0)
    values[np.abs(nearest - coords[:, axis]) > half + 1e-6] = 0
    return values.astype(array.dtype)


def make_stages(config=None, **sections):
    """list of Config objects (one per registration stage)"""
    if isinstance(config, (str, os.PathLike)):
        config = _configfile.read(config)
    if isinstance(config, (list, tuple)):
        if sections:
            raise ValueError("Keyword settings cannot be combined with several stages")
        if not config:
            raise ValueError("No registration stage")
        return [Config(cfg) for cfg in config]
    return [Config(config, **sections)]


def _tolist(value):
    """serializable value: None, scalar or (nested) list"""
    if value is None:
        return None
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value.tolist()


# Decorators


def splitcomplex(narg=1, how="add", **opts):
    """decorator, split and merge (ie add, concatenate) complex valued function"""

    def addvalues(rvalues, ivalues):
        return rvalues + ivalues

    def catvalues(rvalues, ivalues):
        return np.concatenate([rvalues, ivalues], **opts)

    def complexvalues(rvalues, ivalues):
        return rvalues + 1j * ivalues

    joinfuncs = {"add": addvalues, "concatenate": catvalues, "complex": complexvalues}

    how = [how] if isinstance(how, str) else list(how)
    if not set(how) <= set(joinfuncs):
        raise NotImplementedError(how)

    def decorator(classfunc):
        def wrapper(self, *args, **kwargs):
            if not any(np.iscomplexobj(obj) for obj in args[:narg]):
                return classfunc(self, *args, **kwargs)
            rargs = tuple(np.real(arg) for arg in args[:narg]) + args[narg:]
            iargs = tuple(np.imag(arg) for arg in args[:narg]) + args[narg:]
            rvalues = classfunc(self, *rargs, **kwargs)
            ivalues = classfunc(self, *iargs, **kwargs)
            if not isinstance(rvalues, tuple):
                # single return value
                return joinfuncs[how[0]](rvalues, ivalues)
            elif len(rvalues) <= len(how):
                # mutiple return values
                return tuple(
                    joinfuncs[method](rval, ival)
                    for method, rval, ival in zip(how, rvalues, ivalues)
                )
            else:
                raise ValueError(f"Too many return values: {len(rvalues)}")

        wrapper.__name__ = classfunc.__name__
        return wrapper

    return decorator


# Component classes


class Geometry:
    """Image grid: point = origin + direction @ (spacing * coords)

    direction: ITK direction matrix (columns: world directions of the array axes)
    """

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def coords(self):
        """pixel coordinates (unit-less)"""
        return np.reshape(np.indices(self.shape), (self.ndim, -1)).T

    @property
    def points(self):
        """pixel coordinates in image's frame of reference"""
        return self.to_points(self.coords)

    @property
    def indices(self):
        """flat indices"""
        return np.arange(np.prod(self.shape))

    @property
    def center(self):
        """world coordinates of the grid center"""
        return self.to_points((self.shape - 1) / 2)

    @property
    def matrix(self):
        """matrix from unit-less coordinates to points (without origin)"""
        return self.direction * self.spacing

    def __init__(self, shape, spacing=None, origin=None, direction=None):
        self.shape = np.asarray(shape)
        ndim = len(self.shape)
        if spacing is None:
            spacing = [1] * ndim
        if origin is None:
            origin = [0] * ndim
        self.spacing = np.asarray(spacing, dtype=float)
        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.eye(ndim) if direction is None else np.asarray(direction, dtype=float).reshape(ndim, ndim)

    @property
    def is_oblique(self):
        return not np.allclose(self.direction, np.eye(self.ndim))

    def to_points(self, coords):
        """convert set of unit-less coordinates into points"""
        coords = np.asarray(coords)
        if not self.is_oblique:
            return coords * self.spacing + self.origin
        return coords @ self.matrix.T + self.origin

    def to_coords(self, points):
        """convert set of points into unit-less coordinates"""
        points = np.asarray(points)
        if not self.is_oblique:
            return (points - self.origin) / self.spacing
        return (points - self.origin) @ np.linalg.inv(self.matrix).T

    def radius(self, center=None):
        """RMS distance of the grid points to a center (default: grid center), in mm"""
        center = self.center if center is None else np.asarray(center)
        extent = self.shape * self.spacing
        offset = np.linalg.norm(self.center - center)
        return float(np.sqrt(np.sum(extent**2) / 12 + offset**2))

    def coords_jacobian(self):
        """jacobian of `to_coords` (ndim x ndim)"""
        return np.linalg.inv(self.matrix)

    def contains(self, points, margin=0.5):
        """points inside the grid extent (pixel centers +/- `margin` pixel, as ITK's IsInsideBuffer)"""
        coords = self.to_coords(points)
        return np.all((coords >= -margin) & (coords <= self.shape - 1 + margin), axis=1)

    def serialize(self):
        data = {
            "shape": self.shape.tolist(),
            "spacing": self.spacing.tolist(),
            "origin": self.origin.tolist(),
        }
        if self.is_oblique:
            data["direction"] = self.direction.tolist()
        return data

    @classmethod
    def deserialize(cls, data):
        return cls(data["shape"], data.get("spacing"), data.get("origin"), data.get("direction"))


def get_direction(obj, ndim):
    """ITK direction matrix of an image-like object (or None)

    obj.direction: direction matrix (columns: axis directions)
    obj.transform: tomio convention (rows: axis directions)
    """
    direction = getattr(obj, "direction", None)
    if direction is not None:
        return np.asarray(direction, dtype=float).reshape(ndim, ndim)
    transform = getattr(obj, "transform", None)
    if transform is not None and np.size(transform) == ndim**2:
        return np.asarray(transform, dtype=float).reshape(ndim, ndim).T
    return None


class Image:
    """Image container"""

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape

    @property
    def coords(self):
        """pixel coordinates (unit-less)"""
        if self.mask is None:
            return self.geometry.coords
        return np.c_[np.nonzero(self.mask)]

    @property
    def points(self):
        return self.geometry.to_points(self.coords)

    @property
    def indices(self):
        """flat indices"""
        if self.mask is None:
            return self.geometry.indices
        return np.flatnonzero(self.mask)

    @property
    def values(self):
        """flattened values"""
        if self.mask is None:
            return self.array.ravel()
        return self.array[self.mask]

    @property
    def array(self):
        return self._array

    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, mask):
        mask = np.asarray(mask) > 0
        assert mask.shape == self.array.shape
        self._mask = mask

    @property
    def spacing(self):
        return self.geometry.spacing

    @property
    def origin(self):
        return self.geometry.origin

    @property
    def direction(self):
        return self.geometry.direction

    @property
    def transform(self):
        """axis directions (tomio convention: rows), e.g. for `tomio.asvolume(image)`"""
        return self.geometry.direction.T

    @property
    def center(self):
        return self.geometry.center

    def __init__(
        self, im, mask=None, spacing=None, origin=None, dtype=None, label=None, direction=None
    ):
        """init Image object

        im: array, Image, or array-like with spacing/origin and direction (or tomio's transform)
        """
        # array
        array = np.asarray(im, dtype=dtype)

        if spacing is None:
            spacing = getattr(im, "spacing", [1] * array.ndim)
        if origin is None:
            origin = getattr(im, "origin", [0] * array.ndim)
        if direction is None:
            direction = get_direction(im, array.ndim)
        if mask is None and isinstance(im, Image):
            mask = im.mask

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            assert mask.shape == array.shape

        self.geometry = Geometry(array.shape, spacing, origin, direction)
        self.label = label
        self._array = array
        self._mask = mask
        self._coefficients = {}
        self._update_range()

    def gradient_images(self):
        """gradient of the image in world coordinates: one Image per world axis (cached)"""
        if "gradient" not in self._coefficients:
            array = np.abs(self.array) if np.iscomplexobj(self.array) else np.asarray(self.array, dtype=float)
            # derivatives along the array axes (per mm), then world axes: grad = D @ (dI / d(index * spacing))
            derivatives = [
                np.gradient(array, self.spacing[d], axis=d) if array.shape[d] > 1 else np.zeros(array.shape)
                for d in range(array.ndim)
            ]
            world = np.einsum("ij,j...->i...", self.direction, np.stack(derivatives))
            self._coefficients["gradient"] = [
                Image(component, spacing=self.spacing, origin=self.origin, direction=self.direction) for component in world
            ]
        return self._coefficients["gradient"]

    def spline_coefficients(self, order):
        """B-spline interpolation coefficients of the array (cached)"""
        if order not in self._coefficients:
            self._coefficients[order] = bspline_coefficients(self.array, order)
        return self._coefficients[order]

    def _update_range(self):
        """intensity range (of the masked values), used for normalization"""
        values = self.values
        if np.iscomplexobj(values):
            values = np.r_[values.real, values.imag]
        values = values[np.isfinite(values)]
        if not values.size:
            self._min, self._max = 0.0, 1.0
        else:
            self._min, self._max = float(values.min()), float(values.max())

    def __array__(self, dtype=None, copy=None):
        """used by np.array(im) and np.asarray(im)"""
        if copy:
            return np.array(self.array, dtype=dtype)
        return np.asarray(self.array, dtype=dtype)

    def __repr__(self):
        shape = "x".join(map(str, self.shape))
        args = [shape, str(self.array.dtype)]
        if self.label:
            args.append(f"'{self.label}'")
        return f"Image({', '.join(args)})"

    def normalize(self, values=None, tol=1e-8, limits=None):
        """normalize given (or image's own) values using image scale

        limits: (low, high) percentiles of the (masked) values used as range (default: min and max)
        """
        if values is None:
            values = self.values
        vmin, vmax = (self._min, self._max) if limits is None else self.percentiles(limits)
        return (values - vmin) / (vmax - vmin + tol)

    def percentiles(self, limits):
        """percentiles of the (masked, finite) values (cached)"""
        key = ("percentiles", tuple(limits))
        if key not in self._coefficients:
            values = self.values
            if np.iscomplexobj(values):
                values = np.abs(values)
            values = values[np.isfinite(values)]
            self._coefficients[key] = tuple(float(v) for v in np.percentile(values, limits)) if values.size else (0.0, 1.0)
        return self._coefficients[key]

    def update(self, values):
        """update array values (the array is copied: the input array is never modified)"""
        values = np.asarray(values)
        self._array = np.array(self._array, dtype=np.result_type(self._array, values))
        self._array.flat[self.indices] = values
        self._coefficients = {}
        self._update_range()

    def copy(self, arr=None, values=None, **kwargs):
        """copy image and set new values"""
        # init array
        if arr is None:
            # copy array
            arr = np.array(self.array)

        # create new Image object
        im = Image(
            arr,
            mask=kwargs.get("mask", self.mask),
            spacing=kwargs.get("spacing", self.spacing),
            origin=kwargs.get("origin", self.origin),
            direction=kwargs.get("direction", self.direction),
            label=kwargs.get("label", self.label),
        )
        # update values
        if values is not None:
            im.update(values)
        return im


class Pyramid(ABC):
    """Base class for image pyramids"""

    @abstractmethod
    def pyramid(self, *images):
        pass

    def __call__(self, *images: [Image]):
        ndims = set(im.ndim for im in images)
        if not len(ndims) == 1:
            raise ValueError(f"Incompatible image dimensions: {ndims}")
        return self.pyramid(*images)

    def __str__(self):
        return str(type(self).__name__)


class PyramidDummy(Pyramid):
    """Pass-through pyramid (does nothing)"""

    def pyramid(self, *images):
        return [0], [tuple(images)]


class PyramidGaussian(Pyramid):
    """Gaussian image pyramid

    At level k, the target spacing is `scale_ratio**k` times the smallest spacing of all images, the same for all
    the axes that are reduced (isotropic voxels first: axes with a larger spacing are not reduced); the target is
    lowered (for all axes) so that no reduced axis has fewer than `minsize` pixels; axes with at most `minsize`
    pixels are not reduced.
    The reduced grids have the exact target spacing and the same center as the original images.

    scale_ratio: spacing ratio between consecutive levels
    minsize: minimum number of pixels of a reduced axis
    minlevel: finest level used (0: original images)
    maxlevel: coarsest level
    order: interpolation order for downsampling
    sigma: gaussian smoothing in pixels of the original image (None: 0.5 * reduction factor of each axis)
    magnitude: use the magnitude of complex images
    downsample: False: smoothing only, the images keep their grid at all levels (as Elastix's default smoothing
        pyramids); avoids tiny coarse images (e.g. for B-spline transforms, whose coarse levels on a few voxels
        can diverge); with random samples, the cost of an iteration does not depend on the image size.
        None (default): set by the registration: downsampling; for non-global transforms (B-splines), no axis
        is reduced below `MINSIZE_DEFORMABLE` (32) pixels (small images: smoothing only)
    """

    MINSIZE_DEFORMABLE = 32

    def __init__(
        self,
        scale_ratio=2,
        minsize=8,
        minlevel=0,
        maxlevel=3,
        order=1,
        sigma=None,
        magnitude=True,
        downsample=None,
    ):
        self.downsample = downsample
        self.maxlevel = maxlevel
        self.minlevel = minlevel
        self.scale_ratio = scale_ratio
        self.sigma = sigma
        self.order = order  # interpolation
        self.minsize = minsize
        self.magnitude = magnitude

    def reduced_grid(self, image, target):
        """spacing and shape of an image reduced to a target spacing (isotropic for the reduced axes)"""
        shape = image.geometry.shape
        # axes that can be reduced: more than `minsize` pixels, finer than the target
        reducible = (shape > max(self.minsize, 1)) & (image.spacing < target)
        if np.any(reducible):
            # largest common spacing keeping `minsize` pixels along every reduced axis
            caps = (shape - 1) * image.spacing / max(self.minsize - 1, 1)
            target = min(target, np.min(caps[reducible]))
        spacing = np.where(reducible, np.maximum(image.spacing, target), image.spacing)
        newshape = np.floor((shape - 1) * image.spacing / spacing + 1e-8).astype(int) + 1
        return spacing, newshape

    def reduce(self, image, spacing, shape):
        """reduced image, with the same center"""
        ratio = spacing / image.spacing
        # coordinates of the new grid in the original grid
        offset = ((image.geometry.shape - 1) - (shape - 1) * ratio) / 2

        # blur image
        if self.sigma is None:
            sigma = np.where(ratio > 1, 0.5 * ratio, 0)
        else:
            sigma = self.sigma
        array = np.asarray(image.array, dtype=float)
        blurred = gaussian_filter(array, sigma) if np.any(sigma) else array
        if self.downsample is False:
            return Image(blurred, mask=image.mask, spacing=image.spacing, origin=image.origin, direction=image.direction)

        # downsample
        reduced = resample_grid(blurred, offset, ratio, shape, order=self.order)

        # mask
        mask = None
        if image.mask is not None:
            dilated = binary_dilation(image.mask) if np.any(ratio > 1) else image.mask
            mask = resample_grid(dilated.astype(np.uint8), offset, ratio, shape, order=0) > 0

        origin = image.geometry.to_points(offset)
        return Image(reduced, mask=mask, spacing=spacing, origin=origin, direction=image.direction)

    def pyramid(self, *images):
        if self.magnitude:
            images = [im.copy(np.abs(im.array)) for im in images]
        pyramid = [tuple(images)]
        min_spacing = np.min([im.spacing for im in images])

        shapes = [tuple(im.shape) for im in images]
        for level in range(1, self.maxlevel + 1):
            target = min_spacing * self.scale_ratio**level
            grids = [self.reduced_grid(im, target) for im in images]
            newshapes = [tuple(shape) for _, shape in grids]
            if newshapes == shapes:
                # no further reduction
                break
            shapes = newshapes
            # reduce the original images
            pyramid.append(tuple(self.reduce(im, *grid) for im, grid in zip(images, grids)))

        # return levels and reduced images
        nlevel = len(pyramid)
        levels = list(range(nlevel))
        first = min(nlevel - 1, self.minlevel)
        return levels[first::][::-1], pyramid[first::][::-1]


class Transformation:
    """Transform with parameters: maps points of the fixed image to the moving image"""

    def __init__(self, transform, geometry, parameters, center=None):
        """
        transform: Transform object, or type name or serialized transform (dict with a "type" key)
        geometry: output grid (fixed image)
        parameters: transform parameters
        center: rotation center (default: center of `geometry`)
        """
        if isinstance(transform, (str, dict)):
            transform = Transform.deserialize(transform)
        elif not isinstance(transform, Transform):
            raise TypeError(f"Not a Transform: {transform}")

        self.transform = transform
        self.geometry = geometry
        self.parameters = np.asarray(parameters)
        self._center = None if center is None else np.asarray(center, dtype=float)

    @property
    def center(self):
        return self.geometry.center if self._center is None else self._center

    @property
    def is_global(self):
        return self.transform.is_global

    def __repr__(self):
        clsname = type(self.transform).__name__
        if self.parameters.size > 12:
            return f"{clsname}({self.parameters.size} parameters, max |p| = {np.abs(self.parameters).max():.2})"
        values = ", ".join([f"{float(val):.2}" for val in self.parameters])
        return f"{clsname}([{values}])"

    def with_geometry(self, geometry):
        """same transformation, other output grid"""
        return Transformation(self.transform, geometry, self.parameters, center=self.center)

    def serialize(self):
        data = {
            "transform": self.transform.serialize(),
            "geometry": self.geometry.serialize(),
            "parameters": self.parameters.tolist(),
        }
        if self._center is not None:
            data["center"] = self._center.tolist()
        return data

    @classmethod
    def deserialize(cls, data):
        transform = Transform.deserialize(data["transform"])
        geometry = Geometry.deserialize(data["geometry"])
        return cls(transform, geometry, data["parameters"], center=data.get("center"))

    def __call__(self, points=None, **kwargs):
        """transform points (default: points of the output grid)"""
        points = self.geometry.points if points is None else points
        return self.transform(self.parameters, points, center=self.center, **kwargs)

    def affine(self):
        """matrix A and vector t of a global transformation: T(x) = A @ x + t"""
        if not self.is_global:
            raise ValueError(f"Not a global transform: {self.transform}")
        ndim = self.geometry.ndim
        points = self(np.vstack([np.zeros(ndim), np.eye(ndim)]))
        return (points[1:] - points[0]).T, points[0]

    def __array__(self, dtype=None, copy=None):
        """numpy converter"""
        return np.asarray(self.parameters, dtype=dtype)


class TransformationSequence:
    """composition of transformations: T(x) = T_n(...T_2(T_1(x)))"""

    def __init__(self, transformations):
        self.transformations = [as_transformation(trans) for trans in transformations]
        if not self.transformations:
            raise ValueError("Empty sequence of transformations")

    def __len__(self):
        return len(self.transformations)

    def __getitem__(self, index):
        return self.transformations[index]

    def __iter__(self):
        return iter(self.transformations)

    def __repr__(self):
        return f"TransformationSequence({self.transformations})"

    @property
    def geometry(self):
        return self.transformations[-1].geometry

    @property
    def parameters(self):
        return [trans.parameters for trans in self.transformations]

    @property
    def is_global(self):
        return all(trans.is_global for trans in self.transformations)

    def with_geometry(self, geometry):
        return TransformationSequence([trans.with_geometry(geometry) for trans in self.transformations])

    def __call__(self, points=None):
        points = self.geometry.points if points is None else points
        for trans in self.transformations:
            points = trans(points)
        return points

    def affine(self):
        A, t = np.eye(self.geometry.ndim), np.zeros(self.geometry.ndim)
        for trans in self.transformations:
            A_, t_ = trans.affine()
            A, t = A_ @ A, A_ @ t + t_
        return A, t

    def serialize(self):
        return [trans.serialize() for trans in self.transformations]

    @classmethod
    def deserialize(cls, data):
        return cls([Transformation.deserialize(item) for item in data])


def as_transformation(obj):
    """Transformation or TransformationSequence from an object or its serialized form"""
    if isinstance(obj, (Transformation, TransformationSequence)):
        return obj
    elif isinstance(obj, dict):
        return Transformation.deserialize(obj)
    elif isinstance(obj, (list, tuple)):
        sequence = TransformationSequence.deserialize(obj)
        return sequence[0] if len(sequence) == 1 else sequence
    raise TypeError(f"Not a transformation: {obj}")


class Transform(ABC):
    """Base class for geometric transform"""

    # affine transform (with fixed matrix and translation)
    is_global = False

    @abstractmethod
    def transform(self, parameters, points, **kwargs):
        pass

    def settings(self):
        """initialization arguments (serializable)"""
        return {}

    def serialize(self):
        return {"type": component_name("transform", self), **self.settings()}

    @classmethod
    def deserialize(cls, data):
        if isinstance(data, str):
            data = {"type": data}
        data = dict(data)
        transform = get_component("transform", data.pop("type"))
        return transform(**data)

    def parse(self, parameters):
        raise NotImplementedError("'parse' is not implemented for this transform")

    def pack(self, *args, **kwargs):
        raise NotImplementedError("'pack' is not implemented for this transform")

    def __call__(self, parameters, points, **kwargs):
        parameters = np.ravel(parameters)
        points = np.atleast_2d(points)
        return self.transform(parameters, points, **kwargs)

    def init(self, ndim):
        raise NotImplementedError("'init' is not implemented for this transform")

    def setup(self, geometry, level=0):
        """prepare the transform for a fixed image geometry (e.g. a control grid)

        level: pyramid level, relative to the finest level used (0)
        """

    def refine(self, previous, parameters):
        """parameters of this transform equivalent to `parameters` of the transform of the previous level"""
        return parameters

    def with_translation(self, parameters, offset):
        """parameters with the given translation (None if not supported)"""
        if getattr(self, "axis", None) is not None:
            return None
        parameters = np.array(parameters, dtype=float)
        parameters[-len(offset):] = offset
        return parameters

    def scales(self, parameters, radius):
        """parameter scales: one unit of each scaled parameter moves the points by about 1 mm

        radius: typical distance of the points to the center (mm)
        """
        return np.ones(len(parameters))

    def __str__(self):
        return str(type(self).__name__)


class TransformTranslation(Transform):
    """Translation transform"""

    def __init__(self, axis=None, init=None):
        self.axis = axis
        self._init = init

    def parse(self, parameters):
        return {"b": parameters}

    def pack(self, b):
        return b

    is_global = True

    def settings(self):
        return {"axis": _tolist(self.axis), "init": _tolist(self._init)}

    def init(self, ndim):
        if self._init is not None:
            return self._init
        elif self.axis is not None:
            return np.zeros(ndim - 1)
        return np.zeros(ndim)

    def inverse(self, parameters):
        return -parameters

    def transform(self, parameters, points, jac=False, **kwargs):
        if not jac:
            return translation(parameters, points, axis=self.axis)
        return translation(parameters, points, jac=True, axis=self.axis)


class TransformRigid(Transform):
    """Rigid transform

    parameters:
        2d: (phi, bx, by)
        3d: Euler angles and translation (ax, ay, az, bx, by, bz), with R = Rz @ Rx @ Ry (as ITK's Euler3DTransform)
        3d with fixed axis: (phi, b1, b2), rotation about `axis` and translation normal to it
    """

    def __init__(self, axis=None, center=None, init=None):
        self.center = None if center is None else np.asarray(center)
        if np.isscalar(axis) and axis in [0, 1, 2]:
            axis = np.eye(3)[axis]
        elif np.isscalar(axis):
            raise ValueError(axis)
        self.axis = None if axis is None else np.asarray(axis) / np.linalg.norm(axis)
        self._init = None if init is None else np.asarray(init)

    is_global = True

    def settings(self):
        return {"axis": _tolist(self.axis), "center": _tolist(self.center), "init": _tolist(self._init)}

    def init(self, ndim):
        if self._init is not None:
            return self._init
        elif ndim == 2:
            return self.pack(phi=0, b=[0, 0])
        elif ndim == 3 and self.axis is not None:
            return self.pack(phi=0, b=[0, 0])
        elif ndim == 3:
            return self.pack(angles=[0, 0, 0], b=[0, 0, 0])
        raise NotImplementedError(f"Invalid number of dimensions: {ndim}")

    def pack(self, **kwargs):
        ndim = len(kwargs["b"])
        if "angles" in kwargs:
            return np.r_[kwargs["angles"], kwargs["b"]]
        elif ndim == 2:
            return np.r_[kwargs["phi"], kwargs["b"]]
        elif ndim == 3 and self.axis is not None:
            return np.r_[kwargs["phi"], kwargs["b"]]
        raise NotImplementedError(f"Invalid number of dimensions: {ndim}")

    def parse(self, parameters):
        if len(parameters) == 3:
            return {"phi": parameters[0], "b": parameters[1:]}
        elif len(parameters) == 6:
            return {"angles": parameters[:3], "b": parameters[3:]}
        raise NotImplementedError(f"Invalid parameters: {parameters}")

    def transform(self, parameters, points, center=None, jac=False, **kwargs):
        if self.center is not None:
            # use pre defined rotation center
            center = self.center
        elif center is None:
            center = 0

        ndim = points.shape[1]
        if ndim == 2:
            if not jac:
                new = rigidtransform2d(parameters, points - center, jac=False)
                return new + center
            else:
                new, jmat = rigidtransform2d(parameters, points - center, jac=True)
                return new + center, jmat

        elif ndim == 3:
            if not jac:
                new = rigidtransform3d(
                    parameters, points - center, axis=self.axis, jac=False
                )
                return new + center
            else:
                new, jmat = rigidtransform3d(
                    parameters, points - center, axis=self.axis, jac=True
                )
                return new + center, jmat
        else:
            raise NotImplementedError(f"Invalid number of dimensions: {ndim}")

    def scales(self, parameters, radius):
        """rotation angles scaled by the radius, translations unchanged"""
        nrot = 3 if len(parameters) == 6 else 1
        return np.r_[np.full(nrot, radius), np.ones(len(parameters) - nrot)]

    def inverse(self, parameters):
        """parameters of the inverse transform (same center)"""
        if len(parameters) == 3:
            # 2d, or 3d with fixed axis (same formula in the plane normal to the axis)
            return rigidinverse2d(parameters)
        return rigidinverse3d(parameters)


class TransformAffine(Transform):
    """Affine transform"""

    def __init__(self, axis=None, init=None):
        self.axis = axis
        self._init = init

    is_global = True

    def settings(self):
        return {"axis": _tolist(self.axis), "init": _tolist(self._init)}

    def init(self, ndim):
        if self._init is not None:
            return self._init
        elif not ndim in [2, 3]:
            raise NotImplementedError(ndim)
        elif self.axis is not None:
            ndim = 2
        A = np.eye(ndim)
        b = np.zeros(ndim)
        return self.pack(A, b)

    def scales(self, parameters, radius):
        """matrix coefficients scaled by the radius, translations unchanged"""
        nmat = len(self.parse(np.asarray(parameters))["A"].flat)
        return np.r_[np.full(nmat, radius), np.ones(len(parameters) - nmat)]

    def parse(self, parameters):
        ndim = {2: 1, 6: 2, 12: 3}.get(len(parameters), None)
        if ndim is None:
            # compute number of dimensions
            ndim = (int(np.sqrt(4 * len(parameters) + 1)) - 1) // 2
        return {
            "A": parameters[: ndim**2].reshape(ndim, ndim),
            "b": parameters[ndim**2 :],
        }

    def pack(self, A, b):
        return np.r_[A.flat, b]

    def transform(self, parameters, points, jac=False, center=None, **kwargs):
        if center is None:
            center = 0
        if not jac:
            new = affinetransform2d3d(
                parameters, points - center, jac=False, axis=self.axis
            )
            return new + center
        # else
        new, jmat = affinetransform2d3d(
            parameters, points - center, jac=True, axis=self.axis
        )
        return new + center, jmat

    def inverse(self, parameters):
        parsed = self.parse(parameters)
        invA = np.linalg.inv(parsed["A"])
        invb = invA @ parsed["b"]
        return np.r_[invA.ravel(), -invb]


class TransformBSpline(Transform):
    """Free-form deformation: cubic B-spline displacement, T(x) = x + u(x)

    u(x) = sum_k beta3((x - x_k) / h) d_k, on a control grid aligned with the fixed image axes, covering its
    domain (with one control point before and two after along each axis, as ITK's BSplineTransform)

    grid_spacing: control point spacing (mm), scalar or per axis; adjusted so that the grid fits the domain
    schedule: grid spacing factor at pyramid level k: schedule ** k (as Elastix's default GridSpacingSchedule);
        the displacement of each level initializes the (finer) grid of the next level; 1: same grid at all levels
    bending: weight of the bending energy penalty (smoothness regularization, relative to the metric)
    axis: axis of the fixed image along which there is no displacement (e.g. 2: in-plane deformation only, slices
        stay slices, as needed by ROIs drawn on some slices); None: displacement along all axes
    grid: control grid (dict: origin, spacing, shape, direction), set from the fixed image by `setup`
    init: initial parameters (displacements of the control points, mm)

    parameters: displacement components along the fixed image axes (number of components x number of control
        points), in mm
    """

    is_global = False

    def __init__(self, grid_spacing=20.0, schedule=2.0, bending=0.0, axis=None, grid=None, init=None):
        self.axis = None if axis is None else int(axis)
        self.grid_spacing = grid_spacing
        self.schedule = float(schedule)
        self.bending = float(bending)
        self.grid = None if grid is None else {key: np.asarray(value, dtype=float) for key, value in grid.items()}
        if self.grid is not None:
            self.grid["shape"] = self.grid["shape"].astype(int)
        self._init = None if init is None else np.asarray(init, dtype=float)

    def settings(self):
        grid = None if self.grid is None else {key: _tolist(value) for key, value in self.grid.items()}
        return {
            "grid_spacing": _tolist(self.grid_spacing), "schedule": self.schedule, "bending": self.bending,
            "axis": self.axis, "grid": grid, "init": _tolist(self._init),
        }

    def setup(self, geometry, level=0):
        if self.grid is not None:
            return
        ndim = geometry.ndim
        extent = (geometry.shape - 1) * geometry.spacing  # first to last pixel centers
        spacing = np.broadcast_to(np.asarray(self.grid_spacing, dtype=float), (ndim,))
        if self._init is None:
            spacing = spacing * self.schedule**level
        mesh = np.maximum(np.ceil(extent / spacing - 1e-6), 1).astype(int)
        spacing = np.where(extent > 0, extent / mesh, 1.0)
        origin = geometry.origin - geometry.direction @ spacing  # one control point before
        self.grid = {"origin": origin, "spacing": spacing, "shape": mesh + 3, "direction": geometry.direction}

    @property
    def ncontrol(self):
        return int(np.prod(self.grid["shape"]))

    @property
    def components(self):
        """fixed image axes along which there is a displacement"""
        return [a for a in range(len(self.grid["shape"])) if a != self.axis]

    @property
    def basis(self):
        """world directions of the displacement components (ndim x number of components)"""
        return self.grid["direction"][:, self.components]

    def control_points(self):
        """world coordinates of the control points"""
        shape = self.grid["shape"]
        indices = np.reshape(np.indices(shape), (len(shape), -1)).T
        return self.grid["origin"] + (indices * self.grid["spacing"]) @ self.grid["direction"].T

    def refine(self, previous, parameters):
        """coefficients interpolating the displacement of the previous (coarser) grid at the control points"""
        if previous.grid is not None and all(
            np.allclose(previous.grid[key], self.grid[key]) for key in ("origin", "spacing", "shape")
        ):
            return parameters
        shape = tuple(self.grid["shape"])
        disp = previous.displacement(parameters, self.control_points()) @ self.basis
        coeffs = [bspline_coefficients(disp[:, a].reshape(shape), 3) for a in range(disp.shape[1])]
        return np.concatenate([c.ravel() for c in coeffs])

    def init(self, ndim):
        if self.grid is None:
            raise ValueError("TransformBSpline: no control grid (call `setup` with the fixed image geometry)")
        if self._init is not None:
            return self._init
        return np.zeros(len(self.components) * self.ncontrol)

    def grid_coordinates(self, points):
        """control grid coordinates of points (clipped: the 4 taps of each point stay in the grid)"""
        coords = (points - self.grid["origin"]) @ self.grid["direction"] / self.grid["spacing"]
        return np.clip(coords, 1, self.grid["shape"] - 3)

    def weights(self, points):
        """flat control point indices and weights of the points (npoint x 4^ndim), cached for the last
        points array (they do not depend on the parameters)"""
        cached = getattr(self, "_weights", (None, None))
        if cached[0] is points:
            return cached[1]
        result = self._weights_of(points)
        self._weights = (points, result)
        return result

    def _weights_of(self, points):
        coords = self.grid_coordinates(points)
        npoint, ndim = coords.shape
        shape = self.grid["shape"]
        strides = [int(np.prod(shape[d + 1:])) for d in range(ndim)]
        flat = np.zeros((npoint,) + (1,) * ndim, dtype=np.intp)
        weights = np.ones((npoint,) + (1,) * ndim)
        for d in range(ndim):
            first, w, _ = _spline_weights(coords[:, d], 3, shape[d])
            index = first[:, np.newaxis].astype(np.intp) + np.arange(4)
            newshape = [npoint] + [1] * ndim
            newshape[d + 1] = 4
            flat = flat + (index * strides[d]).reshape(newshape)
            weights = weights * w.reshape(newshape)
        return flat.reshape(npoint, -1), weights.reshape(npoint, -1)

    def _displacement(self, coeffs, flat, weights):
        """world displacement from the component coefficients"""
        local = np.stack([(coeffs[a][flat] * weights).sum(axis=1) for a in range(len(coeffs))], axis=1)
        return local @ self.basis.T

    def displacement(self, parameters, points, chunk=2**15):
        """displacement u(x) at points"""
        coeffs = np.reshape(parameters, (len(self.components), -1))
        if len(points) <= chunk:
            return self._displacement(coeffs, *self.weights(points))
        disp = np.empty(points.shape)
        for start in range(0, len(points), chunk):
            index = slice(start, start + chunk)
            disp[index] = self._displacement(coeffs, *self._weights_of(points[index]))
        return disp

    def transform(self, parameters, points, jac=False, **kwargs):
        if not jac:
            return points + self.displacement(parameters, points)
        flat, weights = self.weights(points)
        coeffs = np.reshape(parameters, (len(self.components), -1))
        disp = self._displacement(coeffs, flat, weights)
        return points + disp, SparseJacobian(flat, weights, self.ncontrol, self.basis)

    def penalty(self, parameters, gradient=False):
        """bending energy: weight x mean squared second derivatives of the displacement at the control points"""
        if not self.bending:
            return (0.0, np.zeros(len(parameters))) if gradient else 0.0
        shape = tuple(self.grid["shape"])
        ndim = len(shape)
        ncomp = len(self.components)
        coeffs = np.reshape(parameters, (ncomp,) + shape)
        spacing = self.grid["spacing"]
        norm = self.bending / self.ncontrol
        energy = 0.0
        grad = np.zeros_like(coeffs)
        for i in range(ndim):
            for j in range(i, ndim):
                factor = norm * (1 if i == j else 2) / (spacing[i] * spacing[j]) ** 2
                for a in range(ncomp):
                    diff = _second_difference(coeffs[a], i, j)
                    if diff is None:
                        continue
                    energy += factor * np.sum(diff**2)
                    if gradient:
                        grad[a] += _second_difference_adjoint(2 * factor * diff, i, j, shape)
        return (energy, grad.ravel()) if gradient else energy


def _second_difference(array, i, j):
    """second (i == j) or mixed (i != j) central difference along axes i, j (interior), None if too small"""
    if array.shape[i] < 3 or array.shape[j] < 3:
        return None
    ndim = array.ndim
    if i == j:
        def sl(offset):
            index = [slice(None)] * ndim
            index[i] = slice(1 + offset, array.shape[i] - 1 + offset)
            return tuple(index)
        return array[sl(-1)] - 2 * array[sl(0)] + array[sl(1)]

    def sl(oi, oj):
        index = [slice(None)] * ndim
        index[i] = slice(1 + oi, array.shape[i] - 1 + oi)
        index[j] = slice(1 + oj, array.shape[j] - 1 + oj)
        return tuple(index)
    return (array[sl(1, 1)] - array[sl(1, -1)] - array[sl(-1, 1)] + array[sl(-1, -1)]) / 4


def _second_difference_adjoint(residual, i, j, shape):
    """adjoint of `_second_difference`"""
    out = np.zeros(shape)
    ndim = len(shape)
    if i == j:
        for offset, factor in [(-1, 1), (0, -2), (1, 1)]:
            index = [slice(None)] * ndim
            index[i] = slice(1 + offset, shape[i] - 1 + offset)
            out[tuple(index)] += factor * residual
        return out
    for oi, oj, factor in [(1, 1, 0.25), (1, -1, -0.25), (-1, 1, -0.25), (-1, -1, 0.25)]:
        index = [slice(None)] * ndim
        index[i] = slice(1 + oi, shape[i] - 1 + oi)
        index[j] = slice(1 + oj, shape[j] - 1 + oj)
        out[tuple(index)] += factor * residual
    return out


class SparseJacobian:
    """jacobian of the transformed points w/r to B-spline coefficients: the displacement of point n along the
    world direction basis[:, a] depends on the coefficients (a, flat[n, t]) with weights[n, t]"""

    def __init__(self, flat, weights, ncontrol, basis):
        self.flat, self.weights, self.ncontrol, self.basis = flat, weights, ncontrol, np.asarray(basis)

    def rmatvec(self, gpoints, valid=None):
        """J^T g: gpoints (nvalid x ndim) gradient w/r to the (valid) points -> gradient w/r to the parameters"""
        flat, weights = (self.flat, self.weights) if valid is None else (self.flat[valid], self.weights[valid])
        glocal = gpoints @ self.basis
        grad = [
            np.bincount(flat.ravel(), weights=(weights * glocal[:, a, np.newaxis]).ravel(), minlength=self.ncontrol)
            for a in range(glocal.shape[1])
        ]
        return np.concatenate(grad)


#
# Sampler
class Sampler(ABC):
    """Base class for samplers"""

    @abstractmethod
    def sample(self, image):
        pass

    def update(self, **kwargs):
        pass

    def __call__(self, image: Image):
        return self.sample(image)

    def __str__(self):
        return str(type(self).__name__)


class SamplerFull(Sampler):
    """Sample all pixels"""

    def __init__(self):
        self.image = None
        self._samples = None

    def sample(self, image):
        # same arrays at each call (identify the sample set)
        if image is not self.image:
            self.image = image
            self._samples = image.points, image.values
        return self._samples


class SamplerRandom(Sampler):
    """Sample random pixels"""

    def __init__(self, maxsample=3000):
        self.maxsample = maxsample
        self.image = None
        self._samples = None
        self._rgen = np.random.default_rng(42)  # seed = 42

    def sample(self, image):
        if image is not self.image:
            LOGGER.debug("New image")
            self.image = image
            self._values = self.image.values
            self._points = self.image.points
            self.update()
        # use frozen samples
        return self._samples

    def update(self, **kwargs):
        LOGGER.debug("Resample image")
        nsample = min(self._values.size, self.maxsample)
        indices = self._rgen.choice(self._values.size, nsample, replace=False)
        self._samples = self._points[indices], self._values[indices]


class SamplerRandomCoordinates(Sampler):
    """Sample random values at non grid coordinates

    maxsample: number of samples
    order: interpolation order of the fixed image (default 1: as the default (linear) interpolator of the
        moving image, so that identical images match exactly at the identity)
    thick_ratio: along axes with a spacing >= thick_ratio x the smallest spacing (thick slices),
        samples stay on the grid (interpolating between thick slices gives inaccurate values); None: never
    """

    def __init__(self, maxsample=3000, order=1, thick_ratio=2.0):
        self.maxsample = maxsample
        self.thick_ratio = thick_ratio
        self.image = None
        # interpolator options
        self._interp_opts = {"order": order, "mode": "nearest"}
        self._samples = None
        self._rgen = np.random.default_rng(42)  # seed = 42

    def sample(self, image):
        if image is not self.image:
            LOGGER.debug("New image")
            self.image = image
            self._values = self.image.values
            self._points = self.image.points
            self.update()
        # use frozen samples
        return self._samples

    def update(self, **kwargs):
        LOGGER.debug("Resample image")
        nsample = min(self._values.size, self.maxsample)
        maxcoord = np.array(self.image.shape) - 1
        if self.image.mask is None:
            # uniform in the image domain
            coords = self._rgen.uniform(high=maxcoord, size=(nsample, self.image.ndim))
        else:
            # random pixels of the mask, with a random sub-pixel offset
            pixels = np.c_[np.nonzero(self.image.mask)]
            coords = pixels[self._rgen.choice(len(pixels), nsample, replace=nsample > len(pixels))]
            coords = coords + self._rgen.uniform(-0.5, 0.5, size=coords.shape)
            coords = np.clip(coords, 0, maxcoord)
        coords[:, maxcoord == 0] = 0  # quantize points in single valued dimensions
        if self.thick_ratio is not None:
            spacing = np.asarray(self.image.spacing)
            thick = spacing >= self.thick_ratio * spacing.min()
            coords[:, thick] = np.round(coords[:, thick])
        points = self.image.geometry.to_points(coords)
        order = self._interp_opts["order"]
        coeffs = self.image.spline_coefficients(order)
        values = ndspline_interpolation(coeffs, coords, order=order, mode="nearest", prefiltered=True)
        self._samples = points, values


#
# Interpolator
class Interpolator(ABC):
    """Base class for interpolators"""

    order = 3

    @abstractmethod
    def interpolate(self, coeffs, coords, jac):
        """interpolate at coordinates, from the B-spline coefficients of order `self.order`"""

    def __call__(self, image: Image, points, jac=False):
        """interpolate image at points

        jac:
            False: return values
            "points": also return the gradient of each value w/r to its point (npoint x ndim)
            True: also return the gradient w/r to the points (npoint x npoint x ndim)
            array (npoint x ndim x nparam): jacobian of the points w/r to parameters,
                return the jacobian of the values w/r to the parameters (npoint x nparam)
        """
        coords = image.geometry.to_coords(points)
        dcoords = image.geometry.coords_jacobian()
        coeffs = image.spline_coefficients(self.order)
        if isinstance(jac, np.ndarray):
            # jacobian w/r to the points -> w/r to the coordinates
            jac = np.einsum("ij,njk->nik", dcoords, jac)
            return self.interpolate(coeffs, coords, jac=jac)
        elif isinstance(jac, str) and jac == "points":
            # gradient of each value w/r to its point (npoint x ndim)
            values, grad = self.interpolate(coeffs, coords, jac="points")
            return values, grad @ dcoords
        elif jac:
            values, jacmat = self.interpolate(coeffs, coords, jac=True)
            # gradient w/r to the coordinates -> w/r to the points
            return values, jacmat @ dcoords
        return self.interpolate(coeffs, coords)

    def __str__(self):
        return str(type(self).__name__)


class InterpolatorNearestNeighbor(Interpolator):
    """Nearest neighbor interpolation"""

    order = 0

    @splitcomplex(how=["complex", "concatenate"], axis=0)
    def interpolate(self, coeffs, coords, jac=False):
        if jac:
            raise ValueError(
                f"Cannot compute jacobian with nearest-neighbor interpolation"
            )
        # return values
        return ndspline_interpolation(coeffs, coords, order=0, mode="nearest", prefiltered=True)


class InterpolatorLinear(Interpolator):
    """Linear interpolation

    mode: extension beyond the edges (cf scipy.ndimage); "nearest" keeps the cost continuous for
        samples within half a pixel of the edges (the samples further away are discarded by the Objective)
    """

    order = 1

    def __init__(self, mode="nearest", cval=0):
        self.opts = {"mode": mode, "cval": cval}

    @splitcomplex(how=["complex", "concatenate"], axis=0)
    def interpolate(self, coeffs, coords, jac=False):
        return ndspline_interpolation(coeffs, coords, order=1, jac=jac, prefiltered=True, **self.opts)


class InterpolatorBSpline(Interpolator):
    """general B-Spline interpolation"""

    def __init__(self, order=3, mode="nearest", cval=0):
        self.order = order
        self.opts = {"mode": mode, "cval": cval}

    @splitcomplex(how=["complex", "concatenate"], axis=0)
    def interpolate(self, coeffs, coords, jac=False):
        return ndspline_interpolation(coeffs, coords, order=self.order, jac=jac, prefiltered=True, **self.opts)


#
# Cost function
class Metric(ABC):
    """Base class for cost functions"""

    @abstractmethod
    def measure(self, values1, values2, gradient2):
        pass

    @splitcomplex(narg=2, how=["add", "concatenate"])
    def __call__(self, values1, values2, gradient2=False, **kwargs):
        values1 = np.asarray(values1)
        values2 = np.asarray(values2)
        if not len(values1):
            return np.inf if not gradient2 else (np.inf, np.zeros(values2.shape))
        return self.measure(values1, values2, gradient2, **kwargs)

    def __str__(self):
        return str(type(self).__name__)


class MetricMeanSquares(Metric):
    """Mean square difference cost function"""

    def measure(self, values1, values2, gradient2=False, normalize=False, **kwargs):
        # if callable(normalize):
        # values1, values2 = normalize(values1, values2)

        diff = values1 - values2
        msd = np.mean(diff**2)
        if not gradient2:
            return msd
        # jacobian
        grad2 = (-2 * diff) / len(diff)
        return msd, grad2


class MetricNormalizedGradientFields(Metric):
    """Normalized gradient fields (Haber & Modersitzki 2006): 1 - mean (n_f . n_m)^2, n = grad / sqrt(|grad|^2 + eps^2)

    Multimodal (compares edge directions, independent of the contrast), well suited to free-form deformations.
    eta: edge parameter, eps = eta x mean gradient magnitude of each image (small eta: less bias towards
        transforms increasing the moving gradients, more sensitivity to noise)
    samples mapped outside the moving image count as unaligned (not discarded)
    """

    uses_gradients = True

    def __init__(self, eta=0.01):
        self.eta = eta

    def measure(self, values1, values2, gradient2=False, eps=(1.0, 1.0), **kwargs):
        """values1, values2: gradients of the fixed and moving images at the samples (nsample x ndim)"""
        r1 = np.sqrt(np.sum(values1**2, axis=1) + eps[0] ** 2)
        r2 = np.sqrt(np.sum(values2**2, axis=1) + eps[1] ** 2)
        n1 = values1 / r1[:, np.newaxis]
        dot = np.sum(n1 * values2, axis=1) / r2
        cost = 1 - np.mean(dot**2)
        if not gradient2:
            return cost
        # d(dot) / d(values2) = n1 / r2 - (n1 . values2) values2 / r2^3
        ddot = n1 / r2[:, np.newaxis] - (dot / r2**2)[:, np.newaxis] * values2
        return cost, -2 / len(dot) * dot[:, np.newaxis] * ddot


class MetricMutualInformation(Metric):
    """Mutual information

    nbin: number of histogram bins
    limits: intensity range of each image (percentiles), e.g. (0.5, 99.5); values outside are clipped to the
        first / last bin; default (None): min and max (outliers can squeeze the values into a few bins)
    """

    def __init__(self, nbin=32, limits=None):
        self.nbin = nbin
        self.limits = None if limits is None else tuple(limits)

    def measure(self, values1, values2, gradient2=False, normalize=False, **kwargs):
        """mutual information (negative, to be minimized)

        normalize:
            False: values are already in [0, 1]
            True: normalize each set of values by its own range (gradient: range taken as constant)
            callable: normalize(values1, values2), affine in values2 (e.g. using the images' ranges)
        """
        opts = {"windows": ["spline0", "spline3"], "nbin": self.nbin, **kwargs}
        scale = 1.0
        if callable(normalize):
            # derivative of the (affine) normalization of values2
            ref = normalize(np.zeros(2), np.array([0.0, 1.0]))[1]
            scale = ref[1] - ref[0]
            values1, values2 = normalize(values1, values2)
        elif normalize:
            scale = 1 / (np.ptp(values2) + 1e-8)
            values1, values2 = normalize_values(values1), normalize_values(values2)
        # values out of the histogram range (e.g. spline overshoot): clipped, no gradient
        inside = (values2 >= 0) & (values2 <= 1)
        values1, values2 = np.clip(values1, 0, 1), np.clip(values2, 0, 1)
        if not gradient2:
            return mutualinfo(values1, values2, **opts)
        cost, grad = mutualinfo(values1, values2, gradient2=True, **opts)
        return cost, grad * scale * inside


#
# Optimizer
class Optimizer(ABC):
    """Base class for optimizers"""

    @abstractmethod
    def optimize(self, objective, init, **kwargs):
        pass

    def __call__(self, objective, init, **kwargs):
        return self.optimize(objective, init, **kwargs)

    def __str__(self):
        return str(type(self).__name__)


class OptimizerLBFGS(Optimizer):
    """Limited-memory BFGS with backtracking line search (samples fixed during the optimization)

    The parameters are scaled (one unit moves the points by about 1 mm).

    maxiter: maximum number of iterations
    memory: number of corrections kept
    gtol: stop when the largest gradient component is below gtol
    ftol: stop when the relative cost decrease is below ftol
    xtol: stop when the largest parameter change is below xtol (~ mm)
    maxstep: largest parameter change of the first iteration (~ mm)
    """

    def __init__(self, maxiter=500, memory=10, gtol=1e-8, ftol=1e-7, xtol=1e-3, maxstep=1.0):
        self.opts = {"maxiter": maxiter, "memory": memory, "gtol": gtol, "ftol": ftol, "xtol": xtol, "maxstep": maxstep}

    def optimize(self, objective, init):
        sol, info = lbfgs(lambda x: objective(x, gradient=True), init, **self.opts)
        return sol, OptimResults(**info)


class OptimizerSGD(Optimizer):
    """Stochastic gradient descent with new samples at each iteration (as Elastix's stochastic optimizers)

    Iteration k moves the (scaled) parameters by -gain_k * gradient / max(|gradient|, g0), with
    gain_k = a / (k + 1 + A)^alpha, |.| the largest component and g0 that of the first gradient: no step is larger
    than gain_k (~ mm), and the steps shrink when the gradient becomes smaller than at the start. New samples at each iteration avoid fitting
    the transform to a fixed set of samples (needed for transforms with many parameters, e.g. B-splines);
    fixed number of iterations.

    niter: number of iterations (per pyramid level)
    a, A, alpha: gain sequence (A: stability constant, alpha: decay exponent; 0.602 as in Elastix)
    """

    def __init__(self, niter=256, a=4.0, A=20, alpha=0.602):
        self.niter, self.a, self.A, self.alpha = niter, a, A, alpha

    def optimize(self, objective, init):
        x = np.asarray(init, dtype=float).copy()
        cost, norm = np.nan, None
        for k in range(self.niter):
            if k:
                objective.resample()
            cost, grad = objective(x, gradient=True)
            current = np.max(np.abs(grad))
            if norm is None:
                norm = current
            if not np.isfinite(cost) or max(current, norm) == 0:
                break
            x -= self.a / (k + 1 + self.A) ** self.alpha * grad / max(current, norm)
        return x, OptimResults(fun=cost, nit=k + 1, nfev=k + 1, message="Number of iterations reached", success=True)


class OptimizerAuto(Optimizer):
    """Default optimizer: "lbfgs" for global transforms, "sgd" (new samples at each iteration) otherwise"""

    def select(self, transform):
        return OptimizerLBFGS() if transform.is_global else OptimizerSGD()

    def optimize(self, objective, init):
        transform = getattr(getattr(objective, "objective", objective), "transform")
        return self.select(transform).optimize(objective, init)

    def __str__(self):
        return "OptimizerAuto"


# component registry: section -> {type name: class}
COMPONENTS = {
    "transform": {
        "translation": TransformTranslation,
        "rigid": TransformRigid,
        "affine": TransformAffine,
        "bspline": TransformBSpline,
    },
    "pyramid": {
        "gaussian": PyramidGaussian,
        "none": PyramidDummy,
    },
    "sampler": {
        "full": SamplerFull,
        "random": SamplerRandom,
        "random_coordinates": SamplerRandomCoordinates,
    },
    "interpolator": {
        "nearest": InterpolatorNearestNeighbor,
        "linear": InterpolatorLinear,
        "bspline": InterpolatorBSpline,
    },
    "metric": {
        "mse": MetricMeanSquares,
        "mi": MetricMutualInformation,
        "ngf": MetricNormalizedGradientFields,
    },
    "optimizer": {
        "auto": OptimizerAuto,
        "lbfgs": OptimizerLBFGS,
        "sgd": OptimizerSGD,
    },
}


def get_component(section, name):
    """component class from its type name"""
    if section not in COMPONENTS:
        raise ValueError(f"Unknown section: '{section}' (expected one of: {', '.join(COMPONENTS)})")
    try:
        return COMPONENTS[section][name]
    except (KeyError, TypeError):
        choices = ", ".join(COMPONENTS[section])
        raise ValueError(f"Unknown {section} type: '{name}' (expected one of: {choices})") from None


def component_name(section, obj):
    """type name of a component (class or object)"""
    cls = obj if isinstance(obj, type) else type(obj)
    for name, comptype in COMPONENTS[section].items():
        if comptype is cls:
            return name
    raise ValueError(f"Unregistered {section}: {cls.__name__}")


def component_options(cls):
    """initialization options and their defaults, and whether other options are accepted (**kwargs)"""
    if cls.__init__ is object.__init__:
        return {}, False
    parameters = list(inspect.signature(cls.__init__).parameters.values())[1:]
    options = {par.name: par.default for par in parameters if par.kind in (par.POSITIONAL_OR_KEYWORD, par.KEYWORD_ONLY)}
    return options, any(par.kind == par.VAR_KEYWORD for par in parameters)


class Config:
    """Registration settings, one section per component

    sections:
        transform, pyramid, sampler, interpolator, metric, optimizer: component settings, given as
            - a type name, e.g. "rigid" (default options)
            - a dict with a "type" key and options, e.g. {"type": "mi", "nbin": 32}
            - a dict of options only (the current type is kept), e.g. {"maxsample": 5000}
            - a component object, e.g. TransformRigid(axis=2)
        initialize: {"method": ...}, initial translation of the first stage:
            None, "geometry" (align the image centers) or "moments" (align the centers of mass)
        resample: {"method": ...}, interpolation of the registered image (see `transform`)

    examples:
        Config(transform="rigid", metric="mi")
        Config({"transform": {"type": "rigid", "axis": 2}, "metric": "mi"})
        Config("config.yml")
        Config.describe()  # available types and options
    """

    SECTIONS = tuple(COMPONENTS) + ("initialize", "resample")
    INITIALIZE = (None, "geometry", "moments")

    DEFAULTS = {
        "transform": "rigid",
        "pyramid": "gaussian",
        "sampler": "random_coordinates",
        "interpolator": "linear",
        "metric": "mse",
        "optimizer": "auto",
        "initialize": {"method": None},
        "resample": {"method": None},
    }

    def __init__(self, config=None, **sections):
        if isinstance(config, Config):
            config = config.to_dict()
        elif isinstance(config, (str, os.PathLike)):
            config = _configfile.read(config)
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"Invalid configuration: {config!r}")
        self._sections = {}
        for name, value in self.DEFAULTS.items():
            self[name] = value
        for name, value in {**(config or {}), **sections}.items():
            self[name] = value

    def __getitem__(self, name):
        value = self._sections[name]
        return value if isinstance(value, dict) is False else dict(value)

    def __setitem__(self, name, value):
        if name not in self.SECTIONS:
            raise ValueError(f"Unknown section: '{name}' (expected one of: {', '.join(self.SECTIONS)})")
        if name in ("initialize", "resample"):
            value = {"method": value} if value is None or isinstance(value, str) else dict(value)
            unknown = set(value) - {"method"}
            if unknown:
                raise ValueError(f"Unknown {name} option(s): {', '.join(sorted(unknown))}")
            if name == "initialize" and value.get("method") not in self.INITIALIZE:
                raise ValueError(f"Unknown initialize method: {value['method']} (expected one of: {self.INITIALIZE})")
            self._sections[name] = {**self._sections.get(name, {}), **value}
            return
        if isinstance(value, str):
            value = {"type": value}
        elif not isinstance(value, dict):
            # component object
            component_name(name, value)  # registered type
            self._sections[name] = value
            return
        value = dict(value)
        if "type" not in value:
            # options of the current type
            current = self._sections.get(name)
            if not isinstance(current, dict):
                raise ValueError(f"No type for section '{name}'")
            value = {**current, **value}
        cls = get_component(name, value["type"])
        options, any_option = component_options(cls)
        if not any_option:
            unknown = set(value) - {"type"} - set(options)
            if unknown:
                raise ValueError(
                    f"Unknown option(s) for {name} '{value['type']}': {', '.join(sorted(unknown))}"
                    f" (expected: {', '.join(options) or 'none'})"
                )
        self._sections[name] = value

    def __contains__(self, name):
        return name in self._sections

    def __eq__(self, other):
        return isinstance(other, Config) and self.to_dict() == other.to_dict()

    def to_dict(self):
        """settings as a dict (component objects are kept as is)"""
        return {name: (dict(value) if isinstance(value, dict) else value) for name, value in self._sections.items()}

    def __repr__(self):
        return f"Config({self.to_dict()!r})"

    def __str__(self):
        data = {
            name: (value if isinstance(value, dict) else {"type": component_name(name, value), "object": repr(value)})
            for name, value in self.to_dict().items()
        }
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=None).strip()

    def setup(self, name):
        """instantiate the component of a section"""
        if name not in COMPONENTS:
            raise ValueError(f"Unknown component: '{name}' (expected one of: {', '.join(COMPONENTS)})")
        value = self._sections[name]
        if not isinstance(value, dict):
            # component object: fresh copy (components may hold state, e.g. samplers)
            return copy.deepcopy(value)
        options = dict(value)
        cls = get_component(name, options.pop("type"))
        return cls(**options)

    @classmethod
    def describe(cls):
        """available components and their options (text)"""
        lines = []
        for section, types in COMPONENTS.items():
            lines.append(f"{section}: (default: {cls.DEFAULTS[section]})")
            for name, comptype in types.items():
                doc = (comptype.__doc__ or "").strip().splitlines()
                lines.append(f"    {name}: {doc[0] if doc else comptype.__name__}")
                options, any_option = component_options(comptype)
                for option, default in options.items():
                    default = "" if default is inspect.Parameter.empty else f" = {default!r}"
                    lines.append(f"        {option}{default}")
                if any_option:
                    lines.append("        (other options passed through)")
        lines.append("initialize:")
        lines.append("    method = None  # None, 'geometry' (image centers) or 'moments' (centers of mass)")
        lines.append("resample:")
        lines.append("    method = None  # see `transform`")
        return "\n".join(lines)


class Objective:
    """Objective function to be minimized"""

    def __init__(
        self,
        fixed: Image,
        moving: Image,
        transform: Transform,
        sampler: Sampler,
        metric: Metric,
        interpolator: Interpolator,
        center=None,
        initial=None,
    ):
        """
        center: center of rotation (default: center of the fixed image)
        initial: transformation applied before `transform` (e.g. previous registration stages),
            function of the fixed points
        """
        self.fixed = fixed
        self.moving = moving
        self.transform = transform
        self.sampler = sampler
        self.interpolator = interpolator
        self.metric = metric
        self.center = fixed.geometry.center if center is None else np.asarray(center)
        self.initial = initial
        self._moving_mask = None
        if moving.mask is not None:
            self._moving_mask = Image(
                moving.mask.astype(float), spacing=moving.spacing, origin=moving.origin, direction=moving.direction
            )
        limits = getattr(metric, "limits", None)
        self.normalize = lambda f, m: (fixed.normalize(f, limits=limits), moving.normalize(m, limits=limits))

    def moved_samples(self, samples):
        """sample points through the initial transformation (cached per sample set)"""
        if self.initial is None:
            return samples
        cached = getattr(self, "_moved", (None, None))
        if cached[0] is not samples:
            self._moved = (samples, self.initial(samples))
        return self._moved[1]

    def resample(self):
        """draw new samples (stochastic optimizers)"""
        if getattr(self.sampler, "image", None) is not None:
            self.sampler.update()

    def overlap(self, samples, points1, parameters):
        """samples of the current set that are used: those mapped inside the moving image (pixel centers range) by
        the parameters of the first evaluation (start of the optimization), cached per sample set

        samples: sample points (identify the set), points1: after the initial transform"""
        if getattr(self, "_reference", None) is None:
            self._reference = np.array(parameters, dtype=float)
        cached = getattr(self, "_overlap", (None, None))
        if cached[0] is not samples:
            points2 = self.transform(self._reference, points1, center=self.center)
            self._overlap = (samples, self.moving.geometry.contains(points2, margin=0))
        return self._overlap[1]

    def padded(self, inset, points2, values2, grad2=None):
        """values of the samples of the current set (`inset`, see `overlap`): padding value (moving minimum) and
        zero gradient for the samples that left the moving image

        The sample set is defined at the start of the optimization: otherwise the transform could improve the cost by
        moving badly aligned samples out of the moving image.
        returns: values, samples used [, gradients]
        """
        outside = ~self.moving.geometry.contains(points2)
        values2 = np.where(outside, self.moving._min, values2) if np.any(outside) else values2
        used = inset & np.isfinite(values2)
        if self._moving_mask is not None:
            used &= outside | (InterpolatorNearestNeighbor()(self._moving_mask, points2) > 0)
        if grad2 is None:
            return values2, used
        if np.any(outside):
            grad2 = np.array(grad2)
            npoint = len(points2)
            grad2[:npoint][outside] = 0
            if len(grad2) > npoint:  # complex: imaginary part
                grad2[npoint:][outside] = 0
        return values2, used, grad2

    def valid(self, points2, values2):
        """samples inside the moving image (and its mask), with finite values"""
        valid = self.moving.geometry.contains(points2) & np.isfinite(values2)
        if self._moving_mask is not None:
            valid &= InterpolatorNearestNeighbor()(self._moving_mask, points2) > 0
        return valid

    def _edge_scale(self, image):
        """mean gradient magnitude of an image (for gradient-based metrics)"""
        magnitude = np.sqrt(sum(np.asarray(component.array) ** 2 for component in image.gradient_images()))
        values = magnitude[image.mask] if image.mask is not None else magnitude
        return float(np.mean(values)) or 1.0

    def _call_gradients(self, parameters, gradient=False, **kwargs):
        """objective with a metric of the image gradients (e.g. NGF)"""
        transform_args = {"center": self.center}
        points1, _ = self.sampler(self.fixed)
        points1_moved = self.moved_samples(points1)
        if getattr(self, "_fixed_samples", (None,))[0] is not points1:
            # fixed gradients at the samples (unchanged until the samples change)
            values1 = np.stack([self.interpolator(component, points1) for component in self.fixed.gradient_images()], axis=1)
            eta = getattr(self.metric, "eta", 1.0)
            eps = (eta * self._edge_scale(self.fixed), eta * self._edge_scale(self.moving))
            self._fixed_samples = (points1, values1, eps)
        _, values1, eps = self._fixed_samples
        metric_args = {"eps": eps, **kwargs}
        penalty = getattr(self.transform, "penalty", None)
        components = self.moving.gradient_images()

        # samples outside the moving image are kept with a zero gradient (no alignment): discarding them would
        # let the transform improve the cost by pushing badly aligned samples out of the image
        if not gradient:
            points2 = self.transform(parameters, points1_moved, **transform_args)
            values2 = np.stack([self.interpolator(component, points2) for component in components], axis=1)
            values2[~self.valid(points2, np.sum(values2, axis=1))] = 0
            cost = self.metric(values1, values2, **metric_args)
            return cost + penalty(parameters) if penalty is not None else cost

        points2, jtrans = self.transform(parameters, points1_moved, jac=True, **transform_args)
        interpolated = [self.interpolator(component, points2, jac="points") for component in components]
        values2 = np.stack([values for values, _ in interpolated], axis=1)
        dvalues2 = np.stack([grad for _, grad in interpolated], axis=1)  # npoint x component x ndim
        isvalid = self.valid(points2, np.sum(values2, axis=1))
        values2[~isvalid] = 0
        cost, gmetric = self.metric(values1, values2, gradient2=True, **metric_args)
        gmetric = gmetric[isvalid]
        gpoints = np.einsum("nc,ncd->nd", gmetric, dvalues2[isvalid])
        if isinstance(jtrans, np.ndarray):
            gradient = np.einsum("nd,ndk->k", gpoints, jtrans[isvalid])
        else:
            gradient = jtrans.rmatvec(gpoints, isvalid)
        if penalty is not None:
            pcost, pgrad = penalty(parameters, gradient=True)
            cost, gradient = cost + pcost, gradient + pgrad
        return cost, gradient

    def __call__(self, parameters, gradient=False, **kwargs):
        if getattr(self.metric, "uses_gradients", False):
            return self._call_gradients(parameters, gradient=gradient, **kwargs)
        transform_args = {"center": self.center}
        metric_args = {"normalize": self.normalize, **kwargs}
        points1, values1 = self.sampler(self.fixed)
        samples = points1  # identifies the sample set
        points1 = self.moved_samples(samples)
        penalty = getattr(self.transform, "penalty", None)
        if not gradient:
            points2 = self.transform(parameters, points1, **transform_args)
            values2 = self.interpolator(self.moving, points2)
            values2, isvalid = self.padded(self.overlap(samples, points1, parameters), points2, values2)
            cost = self.metric(values1[isvalid], values2[isvalid], **metric_args)
            if penalty is not None:
                cost = cost + penalty(parameters)
            LOGGER.debug(f"cost: {cost:.4}")
            return cost

        # else, also return gradient
        points2, jtrans = self.transform(
            parameters, points1, jac=True, **transform_args
        )
        values2, grad2 = self.interpolator(self.moving, points2, jac="points")
        values2, isvalid, grad2 = self.padded(self.overlap(samples, points1, parameters), points2, values2, grad2)
        cost, gmetric = self.metric(
            values1[isvalid], values2[isvalid], gradient2=True, **metric_args
        )
        # gradient of the cost w/r to the (valid) moving points
        if np.iscomplexobj(values2):
            npoint, nvalid = len(values2), np.count_nonzero(isvalid)
            gpoints = gmetric[:nvalid, None] * grad2[:npoint][isvalid] + gmetric[nvalid:, None] * grad2[npoint:][isvalid]
        else:
            gpoints = gmetric[:, None] * grad2[isvalid]
        # chain rule through the transform
        if isinstance(jtrans, np.ndarray):
            gradient = np.einsum("nd,ndk->k", gpoints, jtrans[isvalid])
        else:
            gradient = jtrans.rmatvec(gpoints, isvalid)
        if penalty is not None:
            pcost, pgrad = penalty(parameters, gradient=True)
            cost, gradient = cost + pcost, gradient + pgrad
        LOGGER.debug(f"cost: {cost:.4}")
        return cost, gradient

class ScaledObjective:
    """objective function of scaled parameters: x = parameters * scales"""

    def __init__(self, objective, scales):
        self.objective = objective
        self.scales = np.asarray(scales, dtype=float)

    def __call__(self, x, gradient=False, **kwargs):
        parameters = np.asarray(x) / self.scales
        if not gradient:
            return self.objective(parameters, **kwargs)
        cost, grad = self.objective(parameters, gradient=True, **kwargs)
        return cost, grad / self.scales

    def resample(self):
        self.objective.resample()


class Elastix:
    """Main registration object"""

    def __init__(self, config, loglevel=None):
        self.config = config
        self.logger = LOGGER
        loglevel = loglevel if loglevel else "WARN"  # default log-level
        self.logger.setLevel(getattr(logging, str(loglevel), loglevel))

    def execute(self, fixed: Image, moving: Image, init=None, callback=None, initial=None):
        """compute transform

        init: initial parameters
        callback: called after each level: callback(fixed, moving, parameters, config, level=level)
        initial: transformation applied before the optimized transform (previous stages)
        """
        LOGGER.info("Execute pipeline")

        pyramid = self.config.setup("pyramid")
        if getattr(pyramid, "downsample", False) is None:
            # downsampling; for non-global transforms, not below a minimum size (tiny levels can diverge)
            pyramid.downsample = True
            if not self.config.setup("transform").is_global:
                pyramid.minsize = max(pyramid.minsize, pyramid.MINSIZE_DEFORMABLE)
        optimizer = self.config.setup("optimizer")
        self.logger.info(f"Using pyramid: {pyramid}")
        self.logger.info(f"Using optimizer: {optimizer}")

        # compute pyramid
        levels = pyramid(fixed, moving)

        # same center of rotation at all levels
        center = fixed.geometry.center
        radius = fixed.geometry.radius(center)
        parameters = None if init is None else np.asarray(init, dtype=float)
        method = self.config["initialize"].get("method")
        if parameters is None and initial is None and method is not None:
            transform = self.config.setup("transform")
            transform.setup(fixed.geometry)
            parameters = initialize(transform, fixed, moving, method)
        results = []
        previous = None  # transform of the previous level
        for level, images in zip(*levels):
            _fixed, _moving = images

            self.logger.info(f"Optimize level: {level}")
            self.logger.debug(f"Shape: fixed={_fixed.shape}, moving={_moving.shape}")
            self.logger.debug(
                f"Spacing: fixed={_fixed.spacing}, moving={_moving.spacing}"
            )

            transform = self.config.setup("transform")
            # full resolution fixed image (e.g. same center at all levels); level relative to the finest one
            transform.setup(fixed.geometry, level=level - levels[0][-1])
            sampler = self.config.setup("sampler")
            interpolator = self.config.setup("interpolator")
            metric = self.config.setup("metric")

            self.logger.info(f"Using transform: {transform}")
            self.logger.info(f"Using sampler: {sampler}")
            self.logger.info(f"Using interpolator: {interpolator}")
            self.logger.info(f"Using metric: {metric}")

            # objective
            objective = Objective(
                _fixed, _moving, transform, sampler, metric, interpolator, center=center, initial=initial
            )

            # initial parameter values
            if parameters is None:
                init = transform.init(fixed.ndim)
            elif previous is not None:
                init = transform.refine(previous, parameters)
            else:
                init = parameters
            previous = transform
            self.logger.debug(f"Initialization: {init}")

            # optimize scaled parameters (one unit ~ 1 mm displacement)
            scales = transform.scales(init, radius)
            scaled = ScaledObjective(objective, scales)
            parameters, info = optimizer(scaled, np.asarray(init, dtype=float) * scales)
            parameters = parameters / scales

            nsample = len(sampler(_fixed)[0])
            self.logger.debug(f"num samples: {nsample}")
            self.logger.debug(info)
            self.logger.info(f"Parameters: {parameters}")
            results.append(
                OptimResults(
                    level=level, fixed_shape=_fixed.shape, moving_shape=_moving.shape, nsample=nsample,
                    parameters=parameters, **{key: getattr(info, key, None) for key in ("fun", "nit", "nfev", "message", "success")}
                )
            )

            if callback:
                callback(_fixed, _moving, parameters, self.config, level=level)

        transformation = Transformation(transform, fixed.geometry, parameters, center=center)
        self.logger.info(f"Found transformation: {transformation}")
        self.logger.info(f"Done.")
        info.levels = results
        return transformation, info


def initialize(transform, fixed, moving, method):
    """initial parameters: translation aligning the image centers ("geometry") or centers of mass ("moments")"""
    if method == "geometry":
        offset = moving.geometry.center - fixed.geometry.center
    elif method == "moments":
        offset = center_of_mass(moving) - center_of_mass(fixed)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    parameters = transform.with_translation(transform.init(fixed.ndim), offset)
    if parameters is None:
        LOGGER.warning(f"Initialization '{method}' is not supported by {transform}: ignored")
    return parameters


def center_of_mass(image):
    """intensity-weighted center (of the masked values), world coordinates"""
    weights = np.abs(image.values)
    if not np.any(weights):
        return image.geometry.center
    return np.average(image.points, axis=0, weights=weights)


#
# Transform stuff


def normalaxes3d(n):
    """return axes normal to n"""
    if np.isscalar(n) and n in [0, 1, 2]:
        n = np.eye(3)[n]
    i = np.argmax(np.abs(n))
    j = (i + 1) % 3
    n2 = np.zeros(3)
    n2[j] = n[i]
    n2[i] = -n[j]
    n3 = np.cross(n, n2)
    return n2 / np.linalg.norm(n2), n3 / np.linalg.norm(n3)


def translation(parameters, points, axis=None, jac=False):
    """tranlation, with option fixed axis"""
    ndim = points.shape[1]
    if ndim == 3 and axis is not None:
        assert len(parameters) == 2
        nx = np.c_[normalaxes3d(axis)]
        b = nx @ parameters
    else:
        nx = np.eye(ndim)
        b = parameters

    new = points + b
    if not jac:
        return new

    # else, also return jacobian (npoint x ndim x nparam)
    npoint = len(points)
    jmat = np.stack([np.broadcast_to(n, (npoint, ndim)) for n in nx.T], axis=-1)
    return new, jmat


def rigidtransform2d(parameters, points, jac=False):
    """2d transform"""
    phi = parameters[0]
    b = parameters[1:]

    A = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    new = points @ A.T + b

    if not jac:
        return new

    # else, compute jacobian
    dA = np.array([[-np.sin(phi), -np.cos(phi)], [np.cos(phi), -np.sin(phi)]])
    jacA = points @ dA.T
    npoint = len(points)
    jacb1 = np.broadcast_to(np.array([[1, 0]]), (npoint, 2))
    jacb2 = np.broadcast_to(np.array([[0, 1]]), (npoint, 2))
    jac = np.stack([jacA, jacb1, jacb2], axis=2)
    return new, jac


def rigidinverse2d(parameters):
    """Return parameters of inverse transform"""
    phi, b = parameters[0], parameters[1:]
    A_ = np.array([[np.cos(-phi), -np.sin(-phi)], [np.sin(-phi), np.cos(-phi)]])
    b_ = -A_ @ b
    return [-phi, b_[0], b_[1]]


def euler_matrices(angles):
    """rotation matrices about x, y, z"""
    ca, cb, cg = np.cos(angles)
    sa, sb, sg = np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    dRx = np.array([[0, 0, 0], [0, -sa, -ca], [0, ca, -sa]])
    dRy = np.array([[-sb, 0, cb], [0, 0, 0], [-cb, 0, -sb]])
    dRz = np.array([[-sg, -cg, 0], [cg, -sg, 0], [0, 0, 0]])
    return (Rx, Ry, Rz), (dRx, dRy, dRz)


def euler_to_matrix(angles):
    """rotation matrix R = Rz @ Rx @ Ry (ITK's Euler3DTransform)"""
    (Rx, Ry, Rz), _ = euler_matrices(angles)
    return Rz @ Rx @ Ry


def matrix_to_euler(R):
    """Euler angles (ax, ay, az) of a rotation matrix R = Rz @ Rx @ Ry"""
    ax = np.arcsin(np.clip(R[2, 1], -1, 1))
    ay = np.arctan2(-R[2, 0], R[2, 2])
    az = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([ax, ay, az])


def rigidtransform3d(parameters, points, axis=None, jac=False):
    """3d Rigid transform

    parameters:
        axis is None: Euler angles (ax, ay, az) and translation (bx, by, bz), R = Rz @ Rx @ Ry
        axis is set: angle (phi) about `axis`, translation (b1, b2) in the plane normal to `axis`
    """
    parameters = np.asarray(parameters, dtype=float)
    if axis is None:
        assert len(parameters) == 6
        return eulertransform3d(parameters, points, jac=jac)

    assert len(parameters) == 3
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    nx = np.c_[normalaxes3d(n)]
    phi = parameters[0]
    b = nx @ parameters[1:]

    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    dotnX = np.dot(points, n)
    projnX = np.c_[dotnX] * n
    crossnX = np.cross(n, points)
    new = cosphi * points + (1 - cosphi) * projnX + sinphi * crossnX + b

    if not jac:
        return new

    # else, compute jacobian
    jacphi = -sinphi * points + sinphi * projnX + cosphi * crossnX
    npoint = len(points)
    jacb = [np.broadcast_to(vec, (npoint, 3)) for vec in nx.T]
    return new, np.stack([jacphi] + jacb, axis=2)


def eulertransform3d(parameters, points, jac=False):
    """3d rigid transform with Euler angles: R = Rz @ Rx @ Ry, parameters (ax, ay, az, bx, by, bz)"""
    (Rx, Ry, Rz), (dRx, dRy, dRz) = euler_matrices(parameters[:3])
    b = parameters[3:]
    new = points @ (Rz @ Rx @ Ry).T + b
    if not jac:
        return new
    dR = [Rz @ dRx @ Ry, Rz @ Rx @ dRy, dRz @ Rx @ Ry]
    npoint = len(points)
    jacR = [points @ mat.T for mat in dR]
    jacb = [np.broadcast_to(vec, (npoint, 3)) for vec in np.eye(3)]
    return new, np.stack(jacR + jacb, axis=2)


def rigidinverse3d(parameters):
    """Return parameters of inverse 3d rigid transform (Euler angles)"""
    parameters = np.asarray(parameters, dtype=float)
    R = euler_to_matrix(parameters[:3])
    return np.r_[matrix_to_euler(R.T), -R.T @ parameters[3:]]


def affinetransform2d3d(parameters, points, axis=None, jac=False):
    """affine transform 2d/3d"""
    ndim = points.shape[1]
    if not ndim in [2, 3]:
        raise ValueError(f"Only 2d or 3d, not {ndim}d")
    elif ndim == 2:
        assert len(parameters) == 6
        A = np.reshape(parameters[:4], (2, 2))
        b = parameters[4:]
        nx = np.eye(2)
    elif axis is not None:
        assert len(parameters) == 6
        if np.isscalar(axis) and axis in [0, 1, 2]:
            axis = np.eye(3)[axis]
        nx = np.c_[normalaxes3d(axis)]
        _A = np.reshape(parameters[:4], (2, 2))
        _b = parameters[4:]
        A = nx @ _A @ nx.T + np.outer(axis, axis)
        b = nx @ _b
    else:
        assert len(parameters) == 12
        A = np.reshape(parameters[:9], (3, 3))
        b = parameters[9:]
        nx = np.eye(3)

    # transform points
    new = points @ A.T + b

    if not jac:
        return new

    npoint = len(points)
    jacA = [points @ np.outer(n1, n2).T for n1 in nx.T for n2 in nx.T]
    jacb = [np.broadcast_to(n, (npoint, ndim)) for n in nx.T]
    jac = np.stack(jacA + jacb, axis=2)
    return new, jac


#
# Interpolation stuff


def bspline_coefficients(array, order):
    """B-spline interpolation coefficients (mirror boundaries), computed with SimpleITK

    returns a C-contiguous array (the interpolation uses flat indices)
    """
    array = np.asarray(array)
    if np.iscomplexobj(array):
        return bspline_coefficients(array.real, order) + 1j * bspline_coefficients(array.imag, order)
    array = np.asarray(array, dtype=float)
    if order > 1:
        image = sitk.GetImageFromArray(array.T, isVector=False)
        array = sitk.GetArrayFromImage(sitk.BSplineDecomposition(image, int(order))).T
    return np.ascontiguousarray(array)


def bspline_kernel(x, order):
    """centered B-spline of degree `order` (order >= 1)"""
    values = np.zeros(np.shape(x))
    shift = (order + 1) / 2
    for k in range(order + 2):
        values += (-1) ** k * math.comb(order + 1, k) * np.maximum(x + shift - k, 0) ** order
    return values / math.factorial(order)


def bspline_kernel_derivative(x, order):
    """derivative of the centered B-spline of degree `order` (order >= 2)"""
    return bspline_kernel(x + 0.5, order - 1) - bspline_kernel(x - 0.5, order - 1)


def mirror_index(index, size):
    """mirror boundary conditions (whole-sample symmetric, as SimpleITK and scipy's "mirror")"""
    period = np.maximum(2 * size - 2, 1)
    index = np.mod(index, period)
    return np.where(size == 1, 0, np.where(index >= size, period - index, index))


def _spline_weights(coords, order, size):
    """first tap index, weights and derivative weights along one axis (npoint x ntap)"""
    if order == 0:
        first = np.floor(coords + 0.5)
        return first, np.ones((len(coords), 1)), np.zeros((len(coords), 1))
    elif order == 1:
        # linear interpolation in the cell [first, first + 1] (left cell at the last node)
        first = np.clip(np.floor(coords), 0, max(size - 2, 0))
        t = (coords - first)[:, np.newaxis]
        weights = np.hstack([1 - t, t])
        dweights = np.broadcast_to([-1.0, 1.0], weights.shape)
        return first, weights, dweights
    elif order == 3:
        # closed form of the cubic B-spline weights (taps first .. first + 3)
        first = np.floor(coords) - 1
        t = (coords - first - 1)[:, np.newaxis]
        t2, t3 = t * t, t * t * t
        weights = np.hstack([(1 - t) ** 3, 3 * t3 - 6 * t2 + 4, -3 * t3 + 3 * t2 + 3 * t + 1, t3]) / 6
        dweights = np.hstack([-((1 - t) ** 2), 3 * t2 - 4 * t, -3 * t2 + 2 * t + 1, t2]) / 2
        return first, weights, dweights
    first = np.floor(coords - (order - 1) / 2)
    dist = coords[:, np.newaxis] - (first[:, np.newaxis] + np.arange(order + 1))
    return first, bspline_kernel(dist, order), bspline_kernel_derivative(dist, order)


def _spline_evaluate(coeffs, coords, order, gradient=False):
    """values (and gradient) of the spline at coordinates"""
    npoint, ndim = coords.shape
    shape = coeffs.shape
    strides = [int(np.prod(shape[d + 1:])) for d in range(ndim)]
    flat = np.zeros((npoint,) + (1,) * ndim, dtype=np.intp)
    weights, dweights = [], []
    for d in range(ndim):
        first, w, dw = _spline_weights(coords[:, d], order, shape[d])
        ntap = w.shape[1]
        index = mirror_index(first[:, np.newaxis].astype(np.intp) + np.arange(ntap), shape[d])
        newshape = [npoint] + [1] * ndim
        newshape[d + 1] = ntap
        flat = flat + (index * strides[d]).reshape(newshape)
        weights.append(w.reshape(newshape))
        dweights.append(dw.reshape(newshape))

    values = np.take(coeffs.ravel(), flat)
    total = values * math.prod(weights) if ndim > 1 else values * weights[0]
    result = total.reshape(npoint, -1).sum(axis=1)
    if not gradient:
        return result, None
    grad = np.empty((npoint, ndim), dtype=result.dtype)
    for d in range(ndim):
        factors = [dweights[e] if e == d else weights[e] for e in range(ndim)]
        grad[:, d] = (values * math.prod(factors)).reshape(npoint, -1).sum(axis=1)
    return result, grad


def ndspline_interpolation(arr, coords, order=3, jac=False, mode="constant", cval=0, prefiltered=False, chunk=2**14):
    """N-d B-spline interpolation at coordinates (pixel units)

    Parameters
        arr: N1 x N2 x ... Nd array (or its B-spline coefficients if `prefiltered`)
        coords: K x d coordinates
        order: spline order (0: nearest neighbor, 1: linear)
        jac:
            False: return values
            "points": also return the gradient of each value w/r to its coordinates (K x d)
            True: also return the gradient w/r to the coordinates (K x K x d)
            array (K x d x P): jacobian of the coordinates w/r to parameters, return the K x P jacobian
        mode: values beyond the edges (coordinates outside [0, N - 1]):
            "constant": `cval`; "nearest": edge values; "mirror": mirror extension
    """
    if mode not in ("constant", "nearest", "mirror"):
        raise ValueError(f"Invalid mode: {mode}")
    coeffs = np.ascontiguousarray(arr if prefiltered else bspline_coefficients(arr, order))
    if not np.iscomplexobj(coeffs):
        coeffs = coeffs.astype(float, copy=False)
    coords = np.asarray(coords, dtype=float)
    npoint, ndim = coords.shape
    maxcoord = np.array(coeffs.shape) - 1
    outside = (coords < 0) | (coords > maxcoord)
    sign = None
    if mode == "mirror":
        # mirror the coordinates into the grid (the spline is symmetric about the edges)
        period = np.maximum(2 * maxcoord, 1)
        coords = np.mod(coords, period)
        reflected = coords > maxcoord
        coords = np.where(maxcoord == 0, 0, np.where(reflected, period - coords, coords))
        sign = np.where(reflected, -1.0, 1.0)
    else:
        coords = np.clip(coords, 0, maxcoord)

    gradient = jac is not False
    values = np.empty(npoint, dtype=coeffs.dtype)
    grad = np.empty((npoint, ndim), dtype=coeffs.dtype) if gradient else None
    for start in range(0, npoint, chunk):
        index = slice(start, start + chunk)
        values[index], grad_ = _spline_evaluate(coeffs, coords[index], order, gradient=gradient)
        if gradient:
            grad[index] = grad_

    if mode == "constant":
        values[outside.any(axis=1)] = cval
    if not gradient:
        return values
    elif order == 0:
        raise ValueError("Cannot compute jacobian nearest-neighbor mode")

    if mode == "constant":
        grad[outside.any(axis=1)] = 0
    elif mode == "nearest":
        # constant extension
        grad[outside] = 0
    else:
        grad *= sign

    if isinstance(jac, str) and jac == "points":
        # gradient of each value w/r to its coordinates (npoint x ndim)
        return values, grad
    elif isinstance(jac, np.ndarray):
        # jacobian w/r to parameters (npoint x nparam)
        return values, np.einsum("nd,ndk->nk", grad, jac)
    # jacobian w/r to the coordinates of all points (npoint x npoint x ndim)
    jacmat = np.zeros((npoint, npoint, ndim), dtype=grad.dtype)
    indices = np.arange(npoint)
    jacmat[indices, indices, :] = grad
    return values, jacmat


def resample_grid(array, origin, spacing, shape, order=3):
    """resample an array on a regular grid (in pixel coordinates), with SimpleITK (edges: nearest)"""
    image = sitk.GetImageFromArray(np.asarray(array, dtype=float).T, isVector=False)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(n) for n in shape])
    resampler.SetOutputOrigin([float(v) for v in origin])
    resampler.SetOutputSpacing([float(v) for v in spacing])
    resampler.SetOutputDirection(np.eye(np.ndim(array)).ravel().tolist())
    resampler.SetInterpolator(_interp.get_sitk_interpolator(order))
    resampler.SetUseNearestNeighborExtrapolator(True)
    resampler.SetOutputPixelType(sitk.sitkFloat64)
    return sitk.GetArrayFromImage(resampler.Execute(image)).T


def gaussian_filter(array, sigma):
    """gaussian smoothing, sigma in pixels (per axis), with SimpleITK

    recursive filter along each smoothed axis (cost independent of sigma); discrete kernel along axes
    with fewer than 8 pixels (not supported by the recursive filter below 4, boundary effects)
    """
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (np.ndim(array),))
    image = sitk.GetImageFromArray(np.asarray(array, dtype=float).T, isVector=False)
    short = np.zeros(len(sigma))
    for axis, value in enumerate(sigma):
        if value <= 0:
            continue
        elif array.shape[axis] < 8:
            short[axis] = value
            continue
        image = sitk.RecursiveGaussian(image, float(value), False, sitk.RecursiveGaussianImageFilter.ZeroOrder, axis)
    if np.any(short):
        image = sitk.DiscreteGaussian(image, (short**2).tolist(), 64, 0.001, False)
    return sitk.GetArrayFromImage(image).T


def binary_dilation(mask):
    """binary dilation (cross structuring element, radius 1), with SimpleITK"""
    image = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8).T, isVector=False)
    dilated = sitk.BinaryDilate(image, [1] * np.ndim(mask), sitk.sitkCross)
    return sitk.GetArrayFromImage(dilated).T > 0


#
# Metric stuff


class Kernel:
    """kernel functions"""

    def __init__(self, type):
        """select kernel type"""
        try:
            self.func = {
                # "gaussian": self.gaussian,
                "spline0": self.spline0,
                "spline2": self.spline2,
                "spline3": self.spline3,
            }[type]
        except KeyError:
            raise NotImplementedError(type)
        self.order = {
            # "gaussian": 0,
            "spline0": 0,
            "spline2": 2,
            "spline3": 3,
        }[type]

    def __call__(self, x):
        return self.func(x)

    @staticmethod
    def spline0(x):
        """0-order spline window"""
        return 1.0 * ((x >= -0.5) & (x < 0.5))

    @staticmethod
    def spline2(x):
        """2-order spline window"""
        absx = np.abs(x)
        res = np.zeros_like(x)
        mask1 = absx < 0.5
        res[mask1] = 1 / 2 * (1.5 - 2 * x[mask1] ** 2)
        mask2 = (absx < 1.5) & (~mask1)
        res[mask2] = 1 / 2 * (absx[mask2] - 1.5) ** 2
        return res
        # return 1 / 2 * (1.5 - 2 * x ** 2) * (absx < 0.5) + 1 / 2 * (absx - 1.5) ** 2 * (
        #     (absx >= 0.5) & (absx < 1.5)
        # )

    @staticmethod
    def spline3(x):
        """3-order spline window"""
        absx = np.abs(x)
        res = np.zeros_like(x)
        mask1 = absx < 1
        res[mask1] = 1 / 6 * (4 - 6 * x[mask1] ** 2 + 3 * absx[mask1] ** 3)
        mask2 = (absx < 2) & (~mask1)
        res[mask2] = 1 / 6 * (2 - absx[mask2]) ** 3
        return res
        # return 1 / 6 * (4 - 6 * x ** 2 + 3 * absx ** 3) * (absx < 1) + 1 / 6 * (
        #     2 - absx
        # ) ** 3 * ((absx >= 1) & (absx < 2))


def normalize_values(values, tol=1e-8):
    """normalize values between 0 and 1"""
    values = np.asarray(values)
    ptp = np.ptp(values)
    if np.isclose(ptp, 0):
        return np.zeros(values.shape)
    return (values - values.min()) / (ptp + tol)


def parzen2d(
    values1,
    values2,
    nbin=10,
    windows=["spline0", "spline3"],
    normalize=False,
    jac2=False,
):
    """2d kernel density estimation"""
    # normalize values
    if callable(normalize):
        values1, values2 = normalize(values1, values2)
    elif normalize:
        values1 = normalize_values(values1)
        values2 = normalize_values(values2)
    else:
        assert np.all((0 <= values1) & (values1 <= 1))
        assert np.all((0 <= values2) & (values2 <= 1))

    kernel1 = Kernel(windows[0])
    kernel2 = Kernel(windows[1])

    pad1 = -(-kernel1.order // 2)
    bins1 = np.arange(-pad1, nbin + pad1)
    diff1 = bins1[:, np.newaxis] - (nbin - 1) * values1
    window1 = kernel1(diff1)
    window1 = np.r_[
        window1[: pad1 + 1].sum(0, keepdims=True),
        window1[pad1 + 1 : pad1 + nbin - 1],
        window1[pad1 + nbin - 1 :].sum(0, keepdims=True),
    ]

    pad2 = -(-kernel2.order // 2)
    bins2 = np.arange(-pad2, nbin + pad2)
    diff2 = bins2[:, np.newaxis] - (nbin - 1) * values2
    window2 = kernel2(diff2)
    window2 = np.r_[
        window2[: pad2 + 1].sum(0, keepdims=True),
        window2[pad2 + 1 : pad2 + nbin - 1],
        window2[pad2 + nbin - 1 :].sum(0, keepdims=True),
    ]

    nvalue = 1.0 * values1.size
    pdf = np.einsum("ik,jk->ij", window1, window2) / nvalue

    if not jac2:
        # return pdf / nvalue
        return pdf

    elif not windows[1] == "spline3":
        raise NotImplementedError(windows[1])

    # jacobien
    jacmat = Kernel.spline2(diff2 + 0.5) - Kernel.spline2(diff2 - 0.5)
    jacmat = np.r_[
        jacmat[: pad2 + 1].sum(0, keepdims=True),
        jacmat[pad2 + 1 : pad2 + nbin - 1],
        jacmat[pad2 + nbin - 1 :].sum(0, keepdims=True),
    ]
    if jac2 == "split":
        # return jacobian as two separate arrays (to be combined with "ik,jk->ijk")
        jac = (window1, jacmat * (1.0 - nbin) / nvalue)
    else:
        # einsum faster than broadcast (?)
        # jac = window1[:, np.newaxis] * jacmat[np.newaxis] * (1.0 - nbin) / nvalue
        jac = np.einsum("ik,jk->ijk", window1, jacmat * (1.0 - nbin) / nvalue)

    return pdf, jac


def _log_ratio(p12):
    """log(p12 / (p1 p2)), 0 where p12 = 0"""
    p1p2 = np.outer(p12.sum(axis=1), p12.sum(axis=0))
    nonzero = (p12 > 0) & (p1p2 > 0)
    log12 = np.zeros_like(p12)
    log12[nonzero] = np.log(p12[nonzero] / p1p2[nonzero])
    return log12


def mutualinfo_scatter(values1, values2, nbin=32, gradient2=False):
    """mutual information (negative) with zero-order (fixed) and cubic (moving) Parzen windows, values in [0, 1]

    Same result as `mutualinfo(..., windows=["spline0", "spline3"])`, by scatter-add: each sample adds to
    1 x 4 bins of the joint histogram, O(N) instead of O(N nbin).
    """
    nvalue = len(values1)
    # fixed: bin of each value (window [-0.5, 0.5))
    bins1 = np.clip(np.ceil((nbin - 1) * values1 - 0.5), 0, nbin - 1).astype(np.intp)
    # moving: 4 cubic taps, folded into the edge bins
    u2 = (nbin - 1) * values2
    taps = np.floor(u2)[:, np.newaxis] + np.arange(-1, 3)
    dist = taps - u2[:, np.newaxis]
    bins2 = np.clip(taps, 0, nbin - 1).astype(np.intp)
    flat = bins1[:, np.newaxis] * nbin + bins2
    weights = Kernel.spline3(dist)
    p12 = np.bincount(flat.ravel(), weights=weights.ravel(), minlength=nbin * nbin).reshape(nbin, nbin) / nvalue
    log12 = _log_ratio(p12)
    mi = -(p12 * log12).sum()
    if not gradient2:
        return mi
    # d window / d value2 = -(nbin - 1) B3'(dist), B3'(x) = B2(x + 1/2) - B2(x - 1/2)
    dweights = Kernel.spline2(dist + 0.5) - Kernel.spline2(dist - 0.5)
    grad = (nbin - 1) / nvalue * (dweights * log12.ravel()[flat]).sum(axis=1)
    return mi, grad


def mutualinfo(values1, values2, gradient2=False, **kwargs):
    """mutual information metric"""
    windows = list(kwargs.get("windows", ["spline0", "spline3"]))
    if windows == ["spline0", "spline3"] and not kwargs.get("normalize"):
        return mutualinfo_scatter(values1, values2, nbin=kwargs.get("nbin", 10), gradient2=gradient2)

    # estimate joint distribution
    jac2 = "split" if gradient2 else False
    res = parzen2d(values1, values2, jac2=jac2, **kwargs)
    p12, jp12 = res if gradient2 else (res, None)

    # mutual information
    log12 = _log_ratio(p12)
    mi = -(p12 * log12).sum()

    if not gradient2:
        return mi

    # gradient w/r to 2nd argument (values2)
    # grad = - np.einsum("ijk,ij->k", jp12, log12)
    # merge jacobian parts and log12
    grad = -np.einsum("ik,jk,ij->k", *jp12, log12)
    return mi, grad


def lbfgs(func, init, maxiter=100, memory=10, gtol=1e-8, ftol=1e-7, xtol=1e-3, maxstep=1.0, maxls=30):
    """limited-memory BFGS with backtracking (Armijo) line search

    func: returns cost and gradient
    stops when: max |gradient| <= gtol, relative cost decrease <= ftol, or max |step| <= xtol
        (the line search stops backtracking below xtol)
    """
    x = np.asarray(init, dtype=float).copy()
    cost, grad = func(x)
    nfev = 1
    corrections = []  # (s, y, rho)
    res = {"success": False, "message": "Maximum number of iterations reached"}
    nit = 0
    for nit in range(1, maxiter + 1):
        if not np.isfinite(cost):
            res["message"] = "Non-finite cost"
            break
        if np.max(np.abs(grad)) <= gtol:
            res.update(success=True, message="Small gradient")
            break

        # search direction: two-loop recursion
        q = np.array(grad, dtype=float)
        alphas = []
        for s, y, rho in reversed(corrections):
            alpha = rho * (s @ q)
            q -= alpha * y
            alphas.append(alpha)
        if corrections:
            s, y, _ = corrections[-1]
            q *= (s @ y) / (y @ y)
        else:
            q *= maxstep / max(np.max(np.abs(grad)), 1e-300)
        for (s, y, rho), alpha in zip(corrections, reversed(alphas)):
            q += s * (alpha - rho * (y @ q))
        direction = -q
        slope = grad @ direction
        if slope >= 0:
            # not a descent direction: steepest descent, reset memory
            corrections = []
            direction = -grad * maxstep / max(np.max(np.abs(grad)), 1e-300)
            slope = grad @ direction

        # backtracking line search
        step = 1.0
        maxdir = np.max(np.abs(direction))
        accepted = False
        for _ in range(maxls):
            if step * maxdir <= xtol:
                break
            xnew = x + step * direction
            cnew, gnew = func(xnew)
            nfev += 1
            if np.isfinite(cnew) and cnew <= cost + 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            res.update(success=True, message="Small step")
            break

        s, y = xnew - x, gnew - grad
        if s @ y > 1e-12 * (y @ y):
            corrections.append((s, y, 1 / (s @ y)))
            corrections = corrections[-memory:]
        decrease = cost - cnew
        x, cost, grad = xnew, cnew, gnew
        LOGGER.debug(f"iter: {nit}, cost: {cost:3.3}, step: {step}")
        if decrease <= ftol * max(abs(cost), 1e-300):
            res.update(success=True, message="Small cost decrease")
            break

    res.update(nit=nit, nfev=nfev, fun=cost, jac=grad)
    return x, res


class OptimResults(SimpleNamespace):
    """dataclass for optimization results"""

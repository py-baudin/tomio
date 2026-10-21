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
import inspect
import logging
from abc import ABC, abstractmethod
from types import SimpleNamespace

import yaml
import numpy as np
from scipy import ndimage, optimize

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
        registered.append(apply(transformation, im, method=method, label="registered"))
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
        'roi', 'labels': ROI / label map interpolation (global transforms of 3d images only)

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
        new = apply(as_transformation(trans), Image(im, label="moving"), geometry=geometry, method=method, label="transformed")
        transformed.append(new)

    if single:
        transformed = transformed[0]
    return transformed


def apply(transformation, moving, *, geometry=None, method=None, label=None):
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
    elif method in ("roi", "labels"):
        raise NotImplementedError(f"Method '{method}' requires a global transform of a real 3d image")
    else:
        order = _interp.get_order(method)
        interpolator = InterpolatorBSpline(order=order, mode="constant", cval=0)
        values = interpolator(moving, transformation()).reshape(tuple(geometry.shape))
        if not np.issubdtype(moving.array.dtype, np.inexact):
            values = np.rint(values).astype(moving.array.dtype)

    if geometry.ndim == 3:
        return _volume.Volume(values, spacing=geometry.spacing, origin=geometry.origin, transform=geometry.direction.T)
    return Image(values, spacing=geometry.spacing, origin=geometry.origin, direction=geometry.direction, label=label)


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
        self._update_range()

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

    def normalize(self, values=None, tol=1e-8):
        """normalize given (or image's own) values using image scale"""
        if values is None:
            values = self.values
        return (values - self._min) / (self._max - self._min + tol)

    def update(self, values):
        """update array values (the array is copied: the input array is never modified)"""
        values = np.asarray(values)
        self._array = np.array(self._array, dtype=np.result_type(self._array, values))
        self._array.flat[self.indices] = values
        self._update_range()

    def summary(self):
        return {
            "ndim": self.ndim,
            "shape": self.shape,
            "dtype": self.array.dtype,
            "mean": np.mean(self.array),
            "std": np.std(self.array),
            "median": np.median(self.array),
            "pc05": np.percentile(self.array, 5),
            "pc95": np.percentile(self.array, 95),
        }

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

    At level k, the target spacing is `scale_ratio**k` times the smallest spacing of all images;
    axes with a larger spacing are not reduced, and no axis is reduced below `minsize` pixels.
    The reduced grids have the exact target spacing and the same center as the original images.

    scale_ratio: spacing ratio between consecutive levels
    minsize: minimum number of pixels of a reduced axis
    minlevel: finest level used (0: original images)
    maxlevel: coarsest level
    order: interpolation order for downsampling
    sigma: gaussian smoothing in pixels of the original image (None: 0.5 * reduction factor of each axis)
    magnitude: use the magnitude of complex images
    """

    def __init__(
        self,
        scale_ratio=2,
        minsize=8,
        minlevel=0,
        maxlevel=3,
        order=3,
        sigma=None,
        magnitude=True,
    ):
        self.maxlevel = maxlevel
        self.minlevel = minlevel
        self.scale_ratio = scale_ratio
        self.sigma = sigma
        self.order = order  # interpolation
        self.minsize = minsize
        self.magnitude = magnitude

    def reduced_grid(self, image, target):
        """spacing and shape of an image reduced to a target spacing"""
        shape = image.geometry.shape
        spacing = np.maximum(image.spacing, target)
        # no reduction below `minsize` pixels
        minsize = np.minimum(shape, self.minsize)
        maxspacing = np.where(minsize > 1, (shape - 1) * image.spacing / np.maximum(minsize - 1, 1), np.inf)
        spacing = np.where(shape > 1, np.minimum(spacing, np.maximum(maxspacing, image.spacing)), image.spacing)
        newshape = np.floor((shape - 1) * image.spacing / spacing + 1e-8).astype(int) + 1
        return spacing, newshape

    def reduce(self, image, spacing, shape):
        """reduced image, with the same center"""
        ratio = spacing / image.spacing
        # coordinates of the new grid in the original grid
        offset = ((image.geometry.shape - 1) - (shape - 1) * ratio) / 2
        axes = [offset[i] + ratio[i] * np.arange(shape[i]) for i in range(image.ndim)]
        grid = np.meshgrid(*axes, indexing="ij")

        # blur image
        if self.sigma is None:
            sigma = np.where(ratio > 1, 0.5 * ratio, 0)
        else:
            sigma = self.sigma
        array = np.asarray(image.array, dtype=float)
        blurred = ndimage.gaussian_filter(array, sigma=sigma) if np.any(sigma) else array

        # downsample
        reduced = ndimage.map_coordinates(blurred, grid, order=self.order, mode="nearest")

        # mask
        mask = None
        if image.mask is not None:
            dilated = ndimage.binary_dilation(image.mask) if np.any(ratio > 1) else image.mask
            mask = ndimage.map_coordinates(dilated.astype(np.uint8), grid, order=0, mode="nearest") > 0

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

    def sample(self, image):
        return image.points, image.values


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
    """Sample random values at non grid coordinates"""

    def __init__(self, maxsample=3000, order=3):
        self.maxsample = maxsample
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
        points = self.image.geometry.to_points(coords)
        values = ndspline_interpolation(self.image.array, coords, **self._interp_opts)
        self._samples = points, values


#
# Interpolator
class Interpolator(ABC):
    """Base class for interpolators"""

    @abstractmethod
    def interpolate(self, image, points, jac):
        pass

    def __call__(self, image: Image, points, jac=False):
        """interpolate image at points

        jac:
            False: return values
            True: also return the gradient w/r to the points (npoint x npoint x ndim)
            array (npoint x ndim x nparam): jacobian of the points w/r to parameters,
                return the jacobian of the values w/r to the parameters (npoint x nparam)
        """
        coords = image.geometry.to_coords(points)
        dcoords = image.geometry.coords_jacobian()
        if isinstance(jac, np.ndarray):
            # jacobian w/r to the points -> w/r to the coordinates
            jac = np.einsum("ij,njk->nik", dcoords, jac)
            return self.interpolate(image.array, coords, jac=jac)
        elif jac:
            values, jacmat = self.interpolate(image.array, coords, jac=True)
            # gradient w/r to the coordinates -> w/r to the points
            return values, jacmat @ dcoords
        return self.interpolate(image.array, coords)

    def __str__(self):
        return str(type(self).__name__)


class InterpolatorNearestNeighbor(Interpolator):
    """Nearest neighbor interpolation"""

    @splitcomplex(how=["complex", "concatenate"], axis=0)
    def interpolate(self, arr, coords, jac=False):
        if jac:
            raise ValueError(
                f"Cannot compute jacobian with nearest-neighbor interpolation"
            )
        # return values
        return ndspline_interpolation(arr, coords, order=0, mode="nearest")


class InterpolatorLinear(Interpolator):
    """Linear interpolation

    mode: extension beyond the edges (cf scipy.ndimage); "nearest" keeps the cost continuous for
        samples within half a pixel of the edges (the samples further away are discarded by the Objective)
    """

    def __init__(self, mode="nearest", cval=0):
        self.opts = {"mode": mode, "cval": cval}

    @splitcomplex(how=["complex", "concatenate"], axis=0)
    def interpolate(self, arr, coords, jac=False):
        return ndspline_interpolation(arr, coords, order=1, jac=jac, **self.opts)


class InterpolatorBSpline(Interpolator):
    """general B-Spline interpolation"""

    def __init__(self, order=3, mode="nearest", cval=0):
        self.opts = {"order": order, "mode": mode, "cval": cval}

    @splitcomplex(how=["complex", "concatenate"], axis=0)
    def interpolate(self, arr, coords, jac=False):
        return ndspline_interpolation(arr, coords, jac=jac, **self.opts)


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
            return np.inf if not gradient2 else (np.inf, np.array([]))
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


class MetricMutualInformation(Metric):
    """Mutual information"""

    def __init__(self, nbin=32):
        self.nbin = nbin

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


class OptimizerGeneric(Optimizer):
    """Scipy minimizer (other options: see scipy.optimize.minimize)

    resample: draw new samples at each iteration (the cost function is then not deterministic,
        which breaks the line searches of quasi-Newton methods)
    """

    def __init__(self, method=None, jac=True, resample=False, **opts):
        self.opts = {**opts, "method": method, "jac": jac, "args": (jac,)}
        self.resample = resample

    def optimize(self, objective, init):
        # scipy minimize
        callback = objective.callback if self.resample else None
        res = optimize.minimize(objective, init, callback=callback, **self.opts)
        return res.x, res


class OptimizerAdaptiveGradientDescent(Optimizer):
    """Adaptive gradient descent (adagrad), new samples at each iteration"""

    def __init__(self, niter=20, alpha=1e-1, eps=1e-5, rtol=1e-5):
        self.opts = {"niter": niter, "alpha": alpha, "eps": eps, "rtol": rtol}

    def optimize(self, objective, init):
        sol, info = agd(
            objective,
            init,
            callback=objective.callback,
            args={"gradient": True},
            **self.opts,
        )
        return sol, OptimResults(**info)


# component registry: section -> {type name: class}
COMPONENTS = {
    "transform": {
        "translation": TransformTranslation,
        "rigid": TransformRigid,
        "affine": TransformAffine,
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
    },
    "optimizer": {
        "scipy": OptimizerGeneric,
        "agd": OptimizerAdaptiveGradientDescent,
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
        resample: {"method": ...}, interpolation of the registered image (see `transform`)

    examples:
        Config(transform="rigid", metric="mi")
        Config({"transform": {"type": "rigid", "axis": 2}, "metric": "mi"})
        Config("config.yml")
        Config.describe()  # available types and options
    """

    SECTIONS = tuple(COMPONENTS) + ("resample",)

    DEFAULTS = {
        "transform": "rigid",
        "pyramid": "gaussian",
        "sampler": "random_coordinates",
        "interpolator": "linear",
        "metric": "mse",
        "optimizer": "scipy",
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
        if name == "resample":
            value = {"method": value} if value is None or isinstance(value, str) else dict(value)
            unknown = set(value) - {"method"}
            if unknown:
                raise ValueError(f"Unknown resample option(s): {', '.join(sorted(unknown))}")
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
        self.normalize = lambda f, m: (fixed.normalize(f), moving.normalize(m))

    def valid(self, points2, values2):
        """samples inside the moving image (and its mask), with finite values"""
        valid = self.moving.geometry.contains(points2) & np.isfinite(values2)
        if self._moving_mask is not None:
            valid &= InterpolatorNearestNeighbor()(self._moving_mask, points2) > 0
        return valid

    def __call__(self, parameters, gradient=False, **kwargs):
        transform_args = {"center": self.center}
        metric_args = {"normalize": self.normalize, **kwargs}
        points1, values1 = self.sampler(self.fixed)
        if self.initial is not None:
            points1 = self.initial(points1)
        if not gradient:
            points2 = self.transform(parameters, points1, **transform_args)
            values2 = self.interpolator(self.moving, points2)
            isvalid = self.valid(points2, values2)
            cost = self.metric(values1[isvalid], values2[isvalid], **metric_args)
            LOGGER.debug(f"parameters: {parameters}, cost: {cost:.4}")
            return cost

        # else, also return gradient
        points2, jtrans = self.transform(
            parameters, points1, jac=True, **transform_args
        )
        values2, jinterp = self.interpolator(self.moving, points2, jac=jtrans)
        isvalid = self.valid(points2, values2)
        cost, gmetric = self.metric(
            values1[isvalid], values2[isvalid], gradient2=True, **metric_args
        )
        if np.iscomplexobj(values2):
            isvalid = np.r_[isvalid, isvalid]
        gradient = np.dot(gmetric, jinterp[isvalid])
        LOGGER.debug(f"parameters: {parameters}, cost: {cost:.4}, gradient: {gradient}")
        return cost, gradient

    def gradient(self, parameters, **kwargs):
        _, gradient = self(parameters, gradient=True, **kwargs)
        return gradient

    def callback(self, parameters, *args):
        self.sampler.update()  # resample if applicable
        return False


class Elastix:
    """Main registration object"""

    def __init__(self, config, loglevel=None):
        self.config = config
        self.logger = LOGGER
        loglevel = loglevel if loglevel else "WARN"  # default log-level
        self.logger.setLevel(getattr(logging, str(loglevel), loglevel))

    def apply(self, transformation, moving, *, geometry=None, method=None, label=None):
        """resample moving image (see `apply`); default method: config["resample"]"""
        if method is None:
            method = self.config["resample"].get("method")
        return apply(transformation, moving, geometry=geometry, method=method, label=label)

    def execute(self, fixed: Image, moving: Image, init=None, callback=None, initial=None):
        """compute transform

        init: initial parameters
        callback: called after each level: callback(fixed, moving, parameters, config, level=level)
        initial: transformation applied before the optimized transform (previous stages)
        """
        LOGGER.info("Execute pipeline")

        pyramid = self.config.setup("pyramid")
        optimizer = self.config.setup("optimizer")
        self.logger.info(f"Using pyramid: {pyramid}")
        self.logger.info(f"Using optimizer: {optimizer}")

        # compute pyramid
        levels = pyramid(fixed, moving)

        # same center of rotation at all levels
        center = fixed.geometry.center
        parameters = None if init is None else np.asarray(init, dtype=float)
        for level, images in zip(*levels):
            _fixed, _moving = images

            self.logger.info(f"Optimize level: {level}")
            self.logger.debug(f"Shape: fixed={_fixed.shape}, moving={_moving.shape}")
            self.logger.debug(
                f"Spacing: fixed={_fixed.spacing}, moving={_moving.spacing}"
            )

            transform = self.config.setup("transform")
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
            init = transform.init(fixed.ndim) if parameters is None else parameters
            self.logger.debug(f"Initialization: {init}")

            # optimize
            parameters, info = optimizer(objective, init)

            self.logger.debug(f"num samples: {len(sampler(_fixed)[0])}")
            self.logger.debug(info)
            self.logger.info(f"Parameters: {parameters}")

            if callback:
                callback(_fixed, _moving, parameters, self.config, level=level)

        transformation = Transformation(transform, fixed.geometry, parameters, center=center)
        self.logger.info(f"Found transformation: {transformation}")
        self.logger.info(f"Done.")
        return transformation, info


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


def ndspline_interpolation(arr, coords, order=3, jac=False, **kwargs):
    """ND-spline interpolation

    Parameters
        arr: N1xN2x...Nd array
        coords: Kxd coordinates array

    """
    arr = np.asarray(arr, dtype=float)
    coords = np.asarray(coords, dtype=float)
    if order > 1:
        # spline coefficients: computed once for the values and the derivatives
        mode = kwargs.get("mode", "constant")
        coeffs = ndimage.spline_filter(arr, order=order, mode=mode)
        opts = dict(kwargs, order=order, prefilter=False)
    else:
        coeffs = arr
        opts = dict(kwargs, order=order)

    def evaluate(coords):
        return ndimage.map_coordinates(coeffs, coords.T, **opts)

    values = evaluate(coords)
    if jac is False:
        return values

    if order == 0:
        raise ValueError("Cannot compute jacobian nearest-neighbor mode")

    # gradient w/r to the coordinates (npoint x ndim)
    npoint = len(coords)
    ndim = arr.ndim
    grad = np.zeros((npoint, ndim))
    if order == 1:
        # exact: difference of the neighboring values along each axis (right-sided on the grid nodes)
        nearest = {**kwargs, "mode": "nearest"}
        nearest.pop("cval", None)
        for d in range(ndim):
            if arr.shape[d] < 2:
                continue
            coords1 = np.copy(coords)
            coords1[:, d] = np.clip(np.floor(coords[:, d]), 0, arr.shape[d] - 2) + 1
            coords2 = coords1 - np.eye(ndim)[d]
            grad[:, d] = ndimage.map_coordinates(arr, coords1.T, order=1, **nearest)
            grad[:, d] -= ndimage.map_coordinates(arr, coords2.T, order=1, **nearest)
            # outside the grid: constant extension
            grad[(coords[:, d] < 0) | (coords[:, d] > arr.shape[d] - 1), d] = 0
    else:
        # central finite differences on the spline
        step = 1e-3
        for d in range(ndim):
            delta = step * np.eye(ndim)[d]
            grad[:, d] = (evaluate(coords + delta) - evaluate(coords - delta)) / (2 * step)

    if isinstance(jac, np.ndarray):
        # jacobian w/r to parameters (npoint x nparam)
        return values, np.einsum("nd,ndk->nk", grad, jac)
    # jacobian w/r to the coordinates of all points (npoint x npoint x ndim)
    jacmat = np.zeros((npoint, npoint, ndim))
    indices = np.arange(npoint)
    jacmat[indices, indices, :] = grad
    return values, jacmat


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
    def gaussian(x):
        """Gaussian window"""
        return np.exp(-0.5 * x**2 * 2**2) * np.sqrt(0.5 / np.pi * 2**2)

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


def parzen1d(values, nbin=10, window="spline3", normalize=False, jac=False):
    """1d kernel density estimation"""
    if callable(normalize):
        values = normalize(values)
    elif normalize:
        values = normalize_values(values)
    else:
        assert np.all((0 <= values) & (values <= 1))

    kernel = Kernel(window)
    pad = -(-kernel.order // 2)

    # centers = np.arange(0, nbin)
    centers = np.arange(-pad, nbin + pad)
    diff = centers[:, np.newaxis] - (nbin - 1) * values
    dist = kernel(diff).sum(axis=1)
    dist = np.r_[
        dist[: pad + 1].sum(),
        dist[pad + 1 : nbin + pad - 1],
        dist[nbin + pad - 1 :].sum(),
    ]
    norm = len(values)
    assert np.isclose(dist.sum(), norm)

    if not jac:
        return dist / norm
    elif window != "spline3":
        raise NotImplementedError(window)

    # warning: rescaling breaks the jacobian calculation

    # spline2 jacobian
    jac = Kernel.spline2(diff + 0.5) - Kernel.spline2(diff - 0.5)
    jac = np.r_[
        jac[: pad + 1].sum(0, keepdims=True),
        jac[pad + 1 : nbin + pad - 1],
        jac[nbin + pad - 1 :].sum(0, keepdims=True),
    ]
    return dist / norm, -(nbin - 1) * jac / norm


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


def mutualinfo(values1, values2, gradient2=False, **kwargs):
    """mutual information metric"""
    # estimate joint distribution
    jac2 = "split" if gradient2 else False
    res = parzen2d(values1, values2, jac2=jac2, **kwargs)
    p12, jp12 = res if gradient2 else (res, None)

    # marginal distributions
    p1p2 = np.outer(p12.sum(axis=1), p12.sum(axis=0))

    # mutual information
    nonzero = (p12 > 0) & (p1p2 > 0)
    log12 = np.zeros_like(p12)
    log12[nonzero] = np.log(p12[nonzero] / p1p2[nonzero])
    mi = -(p12 * log12).sum()

    if not gradient2:
        return mi

    # gradient w/r to 2nd argument (values2)
    # grad = - np.einsum("ijk,ij->k", jp12, log12)
    # merge jacobian parts and log12
    grad = -np.einsum("ik,jk,ij->k", *jp12, log12)
    return mi, grad


def agd(
    func, init, callback=None, niter=20, alpha=1e-1, eps=1e-5, rtol=1e-5, args={}
):  # gamma=0.9
    """adaptive gradient descent

    Based on 'adagrad' as seen in github.com/antonAce/adaptive-gradient-descent
    """
    x = np.asarray(init, dtype=float)
    G = 0
    scale = 1.0  # step reduction after rejected steps

    # last accepted point
    xbest, cbest, gbest, lr = x, np.inf, None, None

    res = {"success": False, "message": "Maximum number of iterations reached"}
    nit = 0
    for nit in range(1, niter + 1):
        # compute current cost and gradient
        cost, grad = func(x, **args)

        if cost > cbest:
            # rejected step: smaller step from the last accepted point
            scale *= 0.5
            x = xbest - scale * lr * gbest
            continue

        stationary = np.abs(cost - cbest) <= rtol * np.abs(cost)
        xbest, cbest, gbest = x, cost, grad
        LOGGER.debug(f"iter: {nit}, cost: {cost:3.3}, grad: {grad}")
        if stationary:
            res["success"] = True
            res["message"] = "Stationnary cost"
            break

        if callback:
            callback(x)

        # adaptive learning rate
        G = G + grad**2
        lr = alpha / np.sqrt(G + eps)
        x = x - scale * lr * grad

    res["nit"] = nit
    res["fun"] = cbest

    return xbest, res


class OptimResults(SimpleNamespace):
    """dataclass for optimization results"""

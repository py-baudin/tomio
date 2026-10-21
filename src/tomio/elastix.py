""" Implementation of the Elastix registration framework

cf:
- https://en.wikipedia.org/wiki/Elastix_(image_registration)
- https://elastix.lumc.nl/doxygen/classelastix_1_1RandomSampler.html
- https://simpleelastix.readthedocs.io/ParameterMaps.html

TODO:
    - refactor Objective: separate gradient ev. function (less func. calls with some optims)
    - non-downsampling pyramid (only smoothing)
    - OptimizerRobinMonro (a_k = a / (k + A)^alpha), alpha=0.6, A=0.1*maxiter, a=...
    - weighted combination of transforms
    - handle image orientation

"""

import warnings
import yaml
import logging
from abc import ABC, abstractmethod
from types import SimpleNamespace
import numpy as np
from scipy import ndimage, optimize

LOGGER = logging.getLogger(__name__)

""" TODO
- handle one or several images in `register()` and `transform()`
"""


def register(
    fixed, moving, *, config=None, return_transform=False, reset=False, **kwargs
):
    """helper function to register moving images"""

    if not isinstance(moving, (list, tuple)):
        single = True
        moving = [moving]
    else:
        # several moving images
        single = False

    # init config object
    if not isinstance(config, Config):
        config = Config(config)

    fixed = Image(fixed, label="fixed")
    moving = [Image(im, label="moving") for im in moving]

    # get pipeline
    pipeline = Elastix(config, loglevel=kwargs.pop("loglevel", "WARN"))

    # optimize
    init = None
    registered = []
    transformations = []
    for im in moving:
        # optimize
        transformation, info = pipeline.execute(fixed, im, init=init, **kwargs)

        # apply transform
        new = pipeline.apply(transformation, im, label="registered")
        registered.append(new)
        transformations.append(transformation)

        if not reset:
            # use optimized parameters for next registration
            init = transformation.parameters

    if single:
        registered = registered[0]
        transformations = transformations[0]

    # return
    if return_transform:
        return registered, transformations
    return registered


def transform(
    transformations, moving, *, ref=None, interpolation_order=3, config=None, **kwargs
):
    """helper function to transform images"""

    if not isinstance(moving, (list, tuple)):
        single = True
        moving = [moving]
    else:
        # several moving images
        single = False

    if not isinstance(transformations, (tuple, list)):
        transformations = [transformations] * len(moving)
    elif single:
        raise ValueError("A single transformation must be given")
    elif len(transformations) != len(moving):
        raise ValueError("Different numbers of moving images and transformations")

    loglevel = kwargs.pop("loglevel", "WARN")

    # init config object
    if not isinstance(config, Config):
        config = Config(config)

    # interpolation order
    config["ResampleInterpolator"]["order"] = interpolation_order

    # get pipeline
    pipeline = Elastix(config, loglevel=loglevel)

    if ref is not None:
        ref = Image(ref, label="reference").geometry
    moving = [Image(im, label="moving") for im in moving]

    # reference image
    transformed = []
    for trans, im in zip(transformations, moving):
        if not isinstance(trans, Transformation):  # parse transformation
            trans = Transformation.deserialize(trans)
        if ref is not None:
            # select source of geometrical information (spacing, origin, etc.)
            trans = Transformation(trans.transform, ref.geom, trans.parameters)
        # apply transformation
        new = pipeline.apply(trans, im, label="transformed")
        transformed.append(new)

    if single:
        transformed = transformed[0]
    return transformed


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
        return self.origin + (self.spacing * self.shape) / 2

    def __init__(self, shape, spacing=None, origin=None):
        self.shape = np.asarray(shape)
        if spacing is None:
            spacing = [1] * len(shape)
        if origin is None:
            origin = [0] * len(shape)
        self.spacing = np.asarray(spacing)
        self.origin = np.asarray(origin)

    def to_points(self, coords):
        """convert set of unit-less coordinates into points"""
        return coords * self.spacing + self.origin

    def to_coords(self, points):
        """convert set of points into unit-less coordinates"""
        return (points - self.origin) / self.spacing

    def serialize(self):
        return {
            "shape": self.shape.tolist(),
            "spacing": self.spacing.tolist(),
            "origin": self.origin.tolist(),
        }

    @classmethod
    def deserialize(cls, data):
        return cls(data["shape"], data.get("spacing"), data.get("origin"))


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
    def center(self):
        return self.geometry.center

    def __init__(
        self, im, mask=None, spacing=None, origin=None, dtype=None, label=None
    ):
        """init Image object"""
        # array
        array = np.asarray(im, dtype=dtype)

        if spacing is None:
            spacing = getattr(im, "spacing", [1] * array.ndim)
        if origin is None:
            origin = getattr(im, "origin", [0] * array.ndim)

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            assert mask.shape == array.shape

        self.geometry = Geometry(array.shape, spacing, origin)
        self.label = label
        self._array = array
        self._mask = mask
        self._min = min(array.real.min(), array.imag.min())
        self._max = max(array.real.max(), array.imag.max())

    def __array__(self):
        """used by np.array(im) and np.asarray(im)"""
        return self.array

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
        """update array values"""
        self._array = np.asarray(self._array, dtype=values.dtype)
        self._array.flat[self.indices] = values
        self._min = min(self._array.real.min(), self._array.imag.min())
        self._max = max(self._array.real.max(), self._array.imag.max())

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

    def __init__(self, **kwargs):
        pass

    def pyramid(self, *images):
        return [0], [tuple(images)]


class PyramidGaussian(Pyramid):
    """Gaussian image pyramid"""

    def __init__(
        self,
        scale_ratio=1.5,
        minsize=5,
        minlevel=0,
        maxlevel=1000,
        order=3,
        sigma=1,
        magnitude=True,
    ):  # sigma=0.6667):
        self.maxlevel = maxlevel
        self.minlevel = minlevel
        self.scale_ratio = scale_ratio
        self.sigma = sigma
        self.order = order  # interpolation
        self.minsize = minsize
        self.magnitude = magnitude

    def pyramid(self, *images):
        if self.magnitude:
            images = [im.copy(np.abs(im.array)) for im in images]
        pyramid = [tuple(images)]

        for level in range(self.maxlevel):
            previous = pyramid[-1]
            if np.max([im.shape for im in previous]) <= self.minsize:
                break

            # get image spacings
            spacings = np.r_[[im.spacing for im in previous]]

            # minimum spacing target
            min_spacing = np.min(spacings) * self.scale_ratio

            # downsample all images at that level
            newlevel = []
            for im in previous:
                # break if image is sufficiently small

                # new spacing
                spacing = np.maximum(im.spacing, min_spacing)
                dimratio = im.spacing / spacing

                # new shape
                # shape = np.maximum(np.rint(im.shape * dimratio).astype(int), 1)
                shape = np.maximum((im.shape * dimratio + 0.5).astype(int), 1)

                # grid coordinates in each directions
                maxcoord = np.array(im.shape) - 1
                coords = [np.linspace(0, maxcoord[i], shape[i]) for i in range(im.ndim)]
                grid = np.meshgrid(*coords, indexing="ij")

                # blur image
                blurred = ndimage.gaussian_filter(im.array, sigma=self.sigma)

                # downsample
                reduced = ndimage.map_coordinates(blurred, grid, order=self.order)

                # mask
                if im.mask is not None:
                    dilated = ndimage.binary_dilation(im.mask)
                    mask = ndimage.map_coordinates(dilated, grid, order=0) > 0
                else:
                    mask = None

                # store new Image
                newim = Image(reduced, mask=mask, spacing=spacing, origin=im.origin)
                newlevel.append(newim)

            pyramid.append(tuple(newlevel))

        # return levels and reduced images
        nlevel = len(pyramid)
        levels = list(range(nlevel))
        first = min(nlevel - 1, self.minlevel)
        return levels[first::][::-1], pyramid[first::][::-1]


class Transformation:
    """Container for transformation parameters"""

    def __init__(self, transform, geometry, parameters, **kwargs):
        if isinstance(transform, str):
            transform = Transform.deserialize(
                {"Settings": kwargs, "TransformType": transform}
            )
        elif not isinstance(transform, Transform):
            raise TypeError(f"Not a Transform: {transform}")

        self.transform = transform
        self.geometry = geometry
        self.parameters = np.asarray(parameters)

    def __repr__(self):
        clsname = type(self.transform).__name__
        values = ", ".join([f"{float(val):.2}" for val in self.parameters])
        return f"{clsname}([{values}])"

    def serialize(self):
        data = {}
        data["Transform"] = self.transform.serialize()
        data["Geometry"] = self.geometry.serialize()
        data["Parameters"] = self.parameters.tolist()
        return data

    @classmethod
    def deserialize(cls, data):
        transform = Transform.deserialize(data["Transform"])
        geometry = Geometry.deserialize(data["Geometry"])
        parameters = data["Parameters"]
        return cls(transform, geometry, parameters)

    def __call__(self, **kwargs):
        """transform points"""
        points = self.geometry.points
        center = self.geometry.center
        return self.transform(self.parameters, points, center=center, **kwargs)

    def __array__(self):
        """numpy converter"""
        return self.parameters


class Transform(ABC):
    """Base class for geometric transform"""

    @abstractmethod
    def transform(self, parameters, points, **kwargs):
        pass

    @classmethod
    def deserialize(cls, data):
        transformtype = data["TransformType"]
        transform = globals()[transformtype]
        assert issubclass(transform, cls)
        # settings
        settings = data.get("Settings", {})
        if any(isinstance(value, str) for value in settings.values()):
            # fix wrong values in settings
            warnings.warn(
                f"Invalid string values in Transformation.deserialize converted to None"
            )
            settings = {
                key: None if isinstance(value, str) else value
                for key, value in settings.items()
            }
        return transform(**settings)

    def parse(self, parameters):
        raise NotImplementedError("'parse' is not implemented for this transform")

    def pack(self, *args, **kwargs):
        raise NotImplementedError("'pack' is not implemented for this transform")

    def serialize(self):
        raise NotImplementedError("'serialize' is not implemented for this transform")

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

    def serialize(self):
        return {"TransformType": "TransformTranslation"}

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
    """Rigid transform"""

    def __init__(self, axis=None, center=None, init=None):
        self.center = None if center is None else np.asarray(center)
        if axis in [0, 1, 2]:
            axis = np.eye(3)[axis]
        elif np.isscalar(axis):
            raise ValueError(axis)
        self.axis = None if axis is None else np.asarray(axis) / np.linalg.norm(axis)
        self._init = None if init is None else np.asarray(init)

    def serialize(self):
        return {
            "TransformType": "TransformRigid",
            "Settings": {
                "axis": None if self.axis is None else self.axis.tolist(),
                "center": None if self.center is None else self.center.tolist(),
                "init": None if self._init is None else self.init.tolist(),
            },
        }

    def init(self, ndim):
        if self._init is not None:
            return self._init
        elif ndim == 2:
            return self.pack(phi=0, b=[0, 0])
        elif ndim == 3 and self.axis is not None:
            return self.pack(phi=0, b=[0, 0])
        elif ndim == 3:
            return self.pack(phi=0, n=[0, 0, 1], b=[0, 0, 0])
        raise NotImplementedError(f"Invalid number of dimensions: {ndim}")

    def pack(self, **kwargs):
        ndim = len(kwargs["b"])
        if ndim == 2:
            return np.r_[kwargs["phi"], kwargs["b"]]
        elif ndim == 3 and self.axis is not None:
            return np.r_[kwargs["phi"], kwargs["b"]]
        elif ndim == 3:
            return np.r_[kwargs["phi"], kwargs["n"], kwargs["b"]]
        raise NotImplementedError(f"Invalid number of dimensions: {ndim}")

    def parse(self, parameters):
        if len(parameters) == 3:
            return {"phi": parameters[0], "b": parameters[1:]}
        elif len(parameters) == 7:
            return {
                "phi": parameters[0],
                "n": parameters[1:4],
                "b": parameters[4:],
            }
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
        if len(parameters) == 3:
            # return self.inverse2d(parameters)
            return rigidinverse2d(parameters)
        else:
            return rigidinverse3d(parameters, axis=self.axis)
        raise NotImplementedError(f"Invalid number of dimensions: {parameters}")


class TransformAffine(Transform):
    """Affine transform"""

    def __init__(self, axis=None, init=None):
        self.axis = axis
        self._init = init

    def serialize(self):
        return {
            "TransformType": "TransformAffine",
            "Settings": {
                "axis": (
                    self.axis
                    if isinstance(self.axis, (type(None), int))
                    else self.axis.tolist()
                ),
                "init": None if self._init is None else self.init.tolist(),
            },
        }

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
        coords = self._rgen.uniform(
            low=0.1 * maxcoord,
            high=0.9 * maxcoord,
            size=(nsample, self.image.ndim),
        )
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
        arr = 1.0 * image.array
        coords = image.geometry.to_coords(points)
        return self.interpolate(arr, coords, jac=jac)

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
    """Linear interpolation"""

    def __init__(self, mode="constant", cval=0):
        self.opts = {"mode": mode, "cval": cval}

    @splitcomplex(how=["complex", "concatenate"], axis=0)
    def interpolate(self, arr, coords, jac=False):
        return ndspline_interpolation(arr, coords, order=1, jac=jac, **self.opts)


class InterpolatorBSpline(Interpolator):
    """general B-Spline interpolation"""

    def __init__(self, order=3, mode="constant", cval=0):
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

    def measure(self, values1, values2, gradient2=False, **kwargs):
        opts = {"windows": ["spline0", "spline3"], "nbin": self.nbin, **kwargs}
        return mutualinfo(values1, values2, gradient2=gradient2, **opts)


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
    """Scipy minimizer"""

    def __init__(self, method=None, jac=True, **opts):
        self.opts = {**opts, "method": method, "jac": jac, "args": (jac,)}

    def optimize(self, objective, init):
        # scipy minimize
        res = optimize.minimize(
            objective, init, callback=objective.callback, **self.opts
        )
        return res.x, res


class OptimizerAdaptiveGradientDescent(Optimizer):
    def __init__(self, **kwargs):
        self.opts = kwargs

    def optimize(self, objective, init):
        sol, info = agd(
            objective,
            init,
            callback=objective.callback,
            args={"gradient": True},
            **self.opts,
        )
        return sol, OptimResults(**info)


class Config:
    """Base class for elastix pipeline settings"""

    COMPONENTS = {
        "Transform": Transform,
        "Sampler": Sampler,
        "Interpolator": Interpolator,
        "Metric": Metric,
        "Optimizer": Optimizer,
        "Pyramid": Pyramid,
    }

    # defaults
    SETTINGS = """
    # Elastix registration configuration

    # Transform type
    Transform: TransformRigid
    # choices: TransformRigid, TransformTranslation, TransformAffine

    # Multi-scale image pyramid
    Pyramid: PyramidGaussian

    # Sample points during optimization
    Sampler: SamplerRandomCoordinates
    # choices: SamplerFull, SamplerRandom, SamplerRandomCoordinates

    # Interpolate image values during optimization
    Interpolator: InterpolatorLinear
    # choices: InterpolatorLinear, InterpolatorNearestNeighbor, InterpolatorBSpline
    # note: registered images are interpolated using InterpolatorBSpline

    # Image comparison metric for optimization
    Metric: MetricMeanSquares
    # choices: MetricMeanSquares, MetricMutualInformation

    # Local optimization algorithm
    Optimizer: OptimizerGeneric
    # OptimizerAdaptiveGradientDescent 

    
    #
    # Component settings

    PyramidGaussian:
        # minlevel: 0 # min downsampling level
        # maxlevel: 5 # max downsampling level
        # minsize: 5 # minimum size in any axis

    # TransformTranslation:
    # TransformAffine:
    TransformRigid:
        # axis: 2 # fixed axis (for 3d volumes)
        # axis: [0, 0, 1] # idem, more general

    # SamplerRandom:
    # SamplerRandomCoordinates:
        # maxsample: 3000 # max number of random samples
    
    OptimizerGeneric:
        jac: True # use jacobian of cost function (recommanded)
        # method: 'cg', 'BFGS', ... (cf scipy.optimize.minimize)

    # other settings
    ResampleInterpolator: # BSpline interpolator
        order: 3

    """

    def __init__(self, custom=None):
        settings = yaml.safe_load(self.SETTINGS)
        if custom:
            settings.update(custom)
        missing = set(self.COMPONENTS) - set(settings)
        if missing:
            raise ValueError(f"Missing components: {missing}")
        self.settings = settings

    def __getitem__(self, key):
        return self.settings[key]

    def __setitem__(self, key, value):
        self.settings[key] = value

    def __repr__(self):
        return repr(self.settings)

    def __str__(self):
        components = {c: self.settings[c] for c in self.COMPONENTS}
        options = {c: self.settings.get(c) for c in components.values()}
        repr = "# Components\n"
        repr += yaml.dump(components)
        repr += "# Options\n"
        for name in options:
            if not options[name]:
                continue
            repr += yaml.dump({name: options[name]})

        return repr
        # return yaml.dump(self.settings, default_flow_style=None)

    def setup(self, component):
        """instantiate component"""
        if not component in self.COMPONENTS:
            raise ValueError(f"Unknown component: '{component}'")
        compname = self.settings.get(component)
        comptype = globals()[compname]
        if not issubclass(comptype, self.COMPONENTS[component]):
            raise ValueError(f"Invalid {component} type: {compname}")
        params = self.settings.get(compname, {})
        settings = dict(params or {})
        return comptype(**settings)


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
    ):
        self.fixed = fixed
        self.moving = moving
        self.transform = transform
        self.sampler = sampler
        self.interpolator = interpolator
        self.metric = metric
        self.normalize = lambda f, m: (fixed.normalize(f), moving.normalize(m))

    def __call__(self, parameters, gradient=False, **kwargs):
        transform_args = {"center": self.fixed.geometry.center}
        metric_args = {"normalize": self.normalize, **kwargs}
        points1, values1 = self.sampler(self.fixed)
        if not gradient:
            points2 = self.transform(parameters, points1, **transform_args)
            values2 = self.interpolator(self.moving, points2)
            isvalid = np.isfinite(values2)
            cost = self.metric(values1[isvalid], values2[isvalid], **metric_args)
            LOGGER.debug(f"parameters: {parameters}, cost: {cost:.4}")
            return cost

        # else, also return gradient
        points2, jtrans = self.transform(
            parameters, points1, jac=True, **transform_args
        )
        values2, jinterp = self.interpolator(self.moving, points2, jac=jtrans)
        isvalid = np.isfinite(values2)
        cost, gmetric = self.metric(
            values1[isvalid], values2[isvalid], gradient2=True, **metric_args
        )
        if np.iscomplexobj(values2):
            isvalid = np.r_[isvalid, isvalid]
        gradient = np.dot(gmetric, jinterp[isvalid])
        LOGGER.debug(f"parameters: {parameters}, cost: {cost:.4}, gradient: {gradient}")
        return cost, gradient

    def gradient(self, parameters, **kwargs):
        _, gradient = self(parameters, {**kwargs, "gradient": True})
        return gradient

    def callback(self, parameters, *args):
        self.sampler.update()  # resample if applicable
        return False


class Elastix:
    """Main registration object"""

    def __init__(self, config, loglevel=None):
        self.config = config
        self.logger = LOGGER.getChild("elastix")
        loglevel = loglevel if loglevel else "WARN"  # default log-level
        self.logger.setLevel(getattr(logging, str(loglevel), loglevel))

    def apply(self, transformation: Transformation, moving: Image, label=None):
        """generate transformed image"""
        assert isinstance(transformation, Transformation)
        interpolator = InterpolatorBSpline(**self.config["ResampleInterpolator"])

        points2 = transformation()
        values = interpolator(moving, points2)
        geom = transformation.geometry.serialize()
        return Image(
            values.reshape(geom["shape"]),
            spacing=geom["spacing"],
            origin=geom["origin"],
            label=label,
        )

    def execute(self, fixed: Image, moving: Image, init=None, callback=None):
        """compute transform"""
        LOGGER.info("Execute pipeline")

        pyramid = self.config.setup("Pyramid")
        optimizer = self.config.setup("Optimizer")
        self.logger.info(f"Using pyramid: {pyramid}")
        self.logger.info(f"Using optimizer: {optimizer}")

        # compute pyramid
        levels = pyramid(fixed, moving)

        parameters = None
        for level, images in zip(*levels):
            _fixed, _moving = images

            self.logger.info(f"Optimize level: {level}")
            self.logger.debug(f"Shape: fixed={_fixed.shape}, moving={_moving.shape}")
            self.logger.debug(
                f"Spacing: fixed={_fixed.spacing}, moving={_moving.spacing}"
            )

            transform = self.config.setup("Transform")
            sampler = self.config.setup("Sampler")
            interpolator = self.config.setup("Interpolator")
            metric = self.config.setup("Metric")

            self.logger.info(f"Using transform: {transform}")
            self.logger.info(f"Using sampler: {sampler}")
            self.logger.info(f"Using interpolator: {interpolator}")
            self.logger.info(f"Using metric: {metric}")

            # objective
            objective = Objective(
                _fixed, _moving, transform, sampler, metric, interpolator
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

        transformation = Transformation(transform, fixed.geometry, parameters)
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


def rigidtransform3d(parameters, points, axis=None, jac=False):
    """3d Rigid transform"""
    if axis is not None:
        assert len(parameters) == 3  # 4
        n = np.asarray(axis)
        nx = np.c_[normalaxes3d(axis)]
        phi = (parameters[:1],)
        _b = parameters[1:]
        b = nx @ _b
    else:
        assert len(parameters) == 7
        nx = np.eye(3)
        phi, n, b = parameters[:1], parameters[1:4], parameters[4:7]

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
    if axis is not None:
        jacn = []
    else:
        select = np.eye(3)
        jacn = [
            (1 - cosphi) * (np.c_[np.dot(points, sel)] * n + np.c_[dotnX] * sel)
            + sinphi * np.cross(sel, points)
            for sel in select
        ]

    # jacb = [np.outer(np.ones(len(points)), row) for row in np.eye(3)]
    npoint = len(points)
    jacb = [np.broadcast_to(n, (npoint, 3)) for n in nx.T]
    jac = np.stack([jacphi] + jacn + jacb, axis=2)
    return new, jac


def rigidinverse3d(parameters, axis=None):
    """Return parameters of inverse 3d rigid transform"""
    if axis is not None:
        assert len(parameters) == 4
        n = np.asarray(axis)
        phi, b = parameters[:1], parameters[1:4]
    else:
        assert len(parameters) == 7
        phi, n, b = parameters[:1], parameters[1:4], parameters[4:7]
    n = np.asarray(n)
    Nc = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    A = np.cos(phi) * np.eye(3) + (1 - np.cos(phi)) * np.outer(n, n) + np.sin(phi) * Nc
    invA = np.linalg.inv(A)
    if axis is not None:
        return np.r_[-phi, -invA @ b]
    else:
        return np.r_[-phi, n / np.linalg.norm(n) ** 2, -invA @ b]


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
    arr = 1.0 * np.asarray(arr)
    coords = np.asarray(coords)
    values = ndimage.map_coordinates(arr, coords.T, order=order, **kwargs)
    if jac is False:
        return values

    if order == 0:
        raise ValueError("Cannot compute jacobian nearest-neighbor mode")

    # jacobian (npoint x npoint x ndim)
    npoint = len(coords)
    ndim = arr.ndim
    opts = dict(**kwargs, order=order)
    if order == 1:
        opts = dict(opts, cval=0, mode="nearest")
        # exact calculation
        jacvalues = np.zeros((npoint, ndim))
        for d in range(ndim):
            coords1 = np.copy(coords)
            coords1[:, d] = np.round(coords1[:, d] + 0.5)
            coords2 = coords1 - np.eye(ndim)[d]
            jacvalues[:, d] = ndimage.map_coordinates(
                arr, coords1.T, **opts
            ) - ndimage.map_coordinates(arr, coords2.T, **opts)
        if isinstance(jac, np.ndarray):
            # jacmat = (jacvalues[..., np.newaxis] * jac).sum(axis=1)
            jacmat = np.nansum(jacvalues[..., np.newaxis] * jac, axis=1)
        else:
            jacmat = np.zeros((npoint, npoint, ndim), dtype=arr.dtype)
            indices = np.arange(npoint)
            jacmat[indices, indices, :] = jacvalues
    else:
        # finite difference
        select = np.eye(ndim * npoint).reshape(npoint * ndim, npoint, ndim).astype(bool)
        eps = 1e-10
        coords1 = coords + 0.5 * eps * select
        coords2 = coords - 0.5 * eps * select
        jacmat = ndimage.map_coordinates(arr, coords1.reshape(-1, ndim).T, **opts)
        jacmat -= ndimage.map_coordinates(arr, coords2.reshape(-1, ndim).T, **opts)
        jacmat = 1e10 * np.moveaxis(jacmat.reshape(npoint, ndim, npoint), 1, 2)
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
        return 1.0 * (np.abs(x) < 0.5)  # ((x >= -0.5) & (x < 0.5))

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
    log12 = np.ma.log(np.ma.divide(p12, p1p2))
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
    init = np.asarray(init)
    G = 0

    def step(x):
        """gradient step"""
        nonlocal G

        # compute current cost and gradient
        cost, grad = func(x, **args)

        # adaptive learning rate
        G = G + grad**2
        lr = alpha / np.sqrt(G + eps)

        return x - lr * grad, cost, grad

    res = {}
    x = init
    prev = np.inf
    for i in range(niter):
        _x, cost, grad = step(x)

        if cost > prev:
            continue
        x = _x

        if np.abs((cost - prev) / cost) < rtol:
            res["success"] = True
            res["message"] = "Stationnary cost"
            break

        prev = cost
        LOGGER.debug(f"iter: {iter}, cost: {cost:3.3}, grad: {grad}")

        if callback:
            callback(x)

    else:
        res["success"] = False
        res["message"] = "Maximum number of iterations reached"

    res["nit"] = i
    res["fun"] = cost

    return x, res


class OptimResults(SimpleNamespace):
    """dataclass for optimization results"""

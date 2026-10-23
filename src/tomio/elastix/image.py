"""images and grid geometry"""

import numpy as np
import SimpleITK as sitk

from .. import volume as _volume
from .splines import bspline_coefficients, gaussian_filter


def _tolist(value):
    """serializable value: None, scalar or (nested) list"""
    if value is None:
        return None
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value.tolist()


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


def direction_to_transform(direction):
    """tomio `transform` (rows: world directions of the array axes) from an ITK direction matrix (columns)"""
    return np.asarray(direction, dtype=float).T


def transform_to_direction(transform, ndim=3):
    """ITK direction matrix (columns: world directions of the array axes) from a tomio `transform` (rows)"""
    return np.asarray(transform, dtype=float).reshape(ndim, ndim).T


class Geometry:
    """Image grid: point = origin + direction @ (spacing * coords)

    direction: ITK direction matrix (columns: world directions of the array axes); tomio's `volume.Geometry`
        stores its transpose (`transform`): see `to_volume`, `from_volume`
    """

    def to_volume(self):
        """tomio `volume.Geometry` (3d)"""
        return _volume.Geometry(
            shape=tuple(int(n) for n in self.shape), spacing=tuple(self.spacing), origin=tuple(self.origin),
            transform=tuple(map(tuple, direction_to_transform(self.direction))),
        )

    @classmethod
    def from_volume(cls, geometry):
        """from a tomio `volume.Geometry` (or a Volume)"""
        ndim = len(geometry.shape)
        return cls(geometry.shape, geometry.spacing, geometry.origin, transform_to_direction(geometry.transform, ndim))

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
        return transform_to_direction(transform, ndim)
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
        return direction_to_transform(self.geometry.direction)

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


def foreground_mask(image, sigma=1.0, dilation=2, min_fraction=0.1, thick_ratio=2.0):
    """foreground of an image (e.g. the body in MRI): Otsu threshold of the smoothed magnitude, connected
    components of at least `min_fraction` x the largest one, holes filled (in 3D and slice by slice), dilated

    sigma: smoothing (pixels), not along thick axes (spacing >= thick_ratio x the smallest)
    dilation: dilation radius (pixels), not along thick axes

    returns: boolean array
    """
    image = image if isinstance(image, Image) else Image(image)
    if "foreground" in image._coefficients:
        return image._coefficients["foreground"]
    array = np.asarray(image.array)
    array = np.abs(array) if np.iscomplexobj(array) else np.asarray(array, dtype=float)
    ndim = array.ndim
    spacing = np.asarray(image.spacing, dtype=float)
    thick = (spacing >= thick_ratio * spacing.min()) | (np.array(array.shape) < 3)
    smooth = gaussian_filter(np.nan_to_num(array), np.where(thick, 0.0, sigma))

    def to_sitk(values):
        return sitk.GetImageFromArray(np.ascontiguousarray(values.T))

    mask = sitk.OtsuThreshold(to_sitk(smooth), 0, 1)  # above the threshold: 1
    labels = sitk.GetArrayFromImage(sitk.ConnectedComponent(mask)).T
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if sizes.max() == 0:
        result = np.ones(array.shape, dtype=bool)
    else:
        keep = np.flatnonzero(sizes >= min_fraction * sizes.max())
        result = np.isin(labels, keep)
        # holes: in nd, then slice by slice along the thickest axis
        result = sitk.GetArrayFromImage(sitk.BinaryFillhole(to_sitk(result.astype(np.uint8)))).T > 0
        if ndim == 3:
            axis = int(np.argmax(spacing))
            for k in range(array.shape[axis]):
                index = (slice(None),) * axis + (k,)
                plane = sitk.BinaryFillhole(to_sitk(result[index].astype(np.uint8)))
                result[index] = sitk.GetArrayFromImage(plane).T > 0
        radius = [0 if t else int(dilation) for t in thick]
        if any(radius):
            dilated = sitk.BinaryDilate(to_sitk(result.astype(np.uint8)), radius)
            result = sitk.GetArrayFromImage(dilated).T > 0
    image._coefficients["foreground"] = result
    return result


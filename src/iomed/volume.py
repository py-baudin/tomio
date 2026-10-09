
from __future__ import annotations

import pathlib
import numpy as np
import numpy.typing as npt
from . import imagefile

DEFAULT_EXT = '.nii.gz'


# I/O

def write(filename, vol, *, nan_as=None, geometry=None, use_compression=True, compression_level=None):
    """ write Volume

    compression_level: if set (format-dependent, e.g. 1-9 for .nii.gz), implies use_compression
    """
    vol = Volume(vol, **(geometry or {}))

    if nan_as is not None:
        # copy: do not modify the input
        vol = Volume(np.where(np.isnan(vol), nan_as, vol), **vol.header)

    # complex: vector image with 2 components (real, imaginary)
    is_complex = np.iscomplexobj(vol)
    data = np.stack([vol.real, vol.imag], axis=-1) if is_complex else vol

    filename = pathlib.Path(filename)
    if not filename.suffix:
        filename = filename.with_suffix(DEFAULT_EXT)

    wrtopts = dict(spacing=vol.spacing, origin=vol.origin, transform=vol.transform, metadata=vol.metadata, use_compression=use_compression, compression_level=compression_level)
    imagefile.write_image(filename, data, is_vector=is_complex, **wrtopts)


def read(filename, *, nan_as=None, as_complex=False, dtype=None):
    """ read Volume

    as_complex: combine a last axis of size 2 (real, imaginary) into complex values
    dtype: cast data to dtype
    """
    array, header = imagefile.read_image(filename)
    if nan_as is not None:
        array[array==nan_as] = np.nan
    if as_complex and array.ndim > 3 and array.shape[-1] == 2:
        array = array[..., 0] + 1j * array[..., 1]
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return Volume(array, **header)


def read_header(filename) -> dict:
    """ read file header only (no pixel data)

    returns: dict with shape (x, y, z, ..., [components]), spacing, origin, transform, metadata
    """
    return imagefile.read_header(filename)



# utilities

def resample(vol, **kwargs) -> Volume:
    """ resample volume to a new geometry (see `interp.resample`, requires scipy)

    resample(vol, like=ref)
    resample(vol, spacing=(1, 1, 1))
    resample(vol, geometry={...}, method="linear", fill=np.nan)
    """
    from . import interp  # lazy import: scipy is optional
    return interp.resample(vol, **kwargs)


def asvolume(obj, ref=None, **kwargs) -> Volume:
    if ref is not None:
        ref = asvolume(ref, **kwargs)
        return Volume(obj, **ref.header)
    return Volume(obj, **kwargs)


def tovolume(values, mask=None, *, ref=None, **kwargs) -> Volume:
    """create and fill Volume at mask values, optionally using ref's metadata"""
    values = np.asarray(values)
    if mask is None and ref is None:
        return asvolume(values, **kwargs)
    elif ref is None:
        vol = asvolume((0 * mask).astype(values.dtype))
    else:
        vol = asvolume((0 * ref.real).astype(values.dtype))

    if mask is not None:
        mask = np.asarray(mask) > 0

    vol[mask] = values
    return vol


class Volume(np.ndarray):
    """ 3D Volume class

    ndarray + spacing/origin/transform/metadata

    bits stolen from https://gitlab.inria.fr/ncedilni/metaio
    """

    def __new__(
        cls,
        data: npt.NDArray[np.generic],
        *,
        spacing: npt.NDArray[np.floating] | None = None,
        origin: npt.NDArray[np.floating] | None = None,
        transform: npt.NDArray[np.floating] | None = None,
        metadata: dict[str, str] | None = None,
        **kwargs, 
    ) -> Volume:
        
        array = np.asanyarray(data, **kwargs).view(cls)
        if spacing is not None:
            array.spacing = spacing
        if origin is not None:
            array.origin = origin
        if transform is not None:
            array.transform = transform
        if metadata is not None:
            array.metadata = metadata
        return array
        
    @staticmethod
    def _get_volume_attrs(obj):
        return dict(
            spacing=getattr(obj, 'spacing', None),
            origin=getattr(obj, 'origin', None),
            transform=getattr(obj, 'transform', None),
            metadata=getattr(obj, 'metadata', None),
        )
    
    @staticmethod
    def _set_volume_attrs(vol, attrs):
        vol.spacing = attrs['spacing']
        vol.origin = attrs['origin']
        vol.transform = attrs['transform']
        vol.metadata = attrs['metadata']
        
    def __array_finalize__(self, obj):
        if obj is None:
            return
        # copy metadata
        self._set_volume_attrs(self, self._get_volume_attrs(obj))
    
    def __getitem__(self, key):
        """ geometry-aware slicing

        slices (incl. negative steps) on the spatial axes: geometry updated
        other indexing on the spatial axes (integer, array, mask, None): plain ndarray
        integer index on the non-spatial axes (>= 3): geometry unchanged
        """
        result = super().__getitem__(key)
        if not isinstance(result, Volume) or self.ndim < 3:
            return result
        index = get_spatial_slices(key, self.shape)
        if index is None:
            return result.view(np.ndarray)
        starts, steps = index
        result.origin = self.get_coords(starts)
        result.spacing = [spacing * abs(step) for spacing, step in zip(self.spacing, steps)]
        result.transform = [[v * np.sign(step) for v in vec] for vec, step in zip(self.transform, steps)]
        return result

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """ numpy ufuncs: compute on plain arrays, then set the header of the result

        elementwise (incl. accumulate): header of the first Volume input with the same shape
        reductions over non-spatial axes only (>= 3): header of the input
        other cases (reductions over spatial axes, shape changes): plain ndarray
        """
        args = [x.view(np.ndarray) if isinstance(x, Volume) else x for x in inputs]
        if out is not None:
            kwargs["out"] = tuple(x.view(np.ndarray) if isinstance(x, Volume) else x for x in out)
        results = getattr(ufunc, method)(*args, **kwargs)

        if method == "at":
            return None
        elif out is not None:
            # results written in `out`
            return out[0] if len(out) == 1 else out
        elif ufunc.nout == 1:
            return self._wrap_ufunc_result(results, method, inputs, kwargs)
        return tuple(self._wrap_ufunc_result(res, method, inputs, kwargs) for res in results)

    @staticmethod
    def _wrap_ufunc_result(result, method, inputs, kwargs):
        if not isinstance(result, np.ndarray) or result.ndim == 0:
            return result
        if method == "reduce":
            return inputs[0]._wrap_reduction(result, kwargs.get("axis", 0))
        elif method in ("__call__", "accumulate"):
            for vol in inputs:
                if isinstance(vol, Volume) and vol.shape == result.shape:
                    return Volume(result, **vol.header)
        return result

    def _wrap_reduction(self, result, axis):
        """ result of a reduction of self over `axis`: Volume if only non-spatial axes are reduced """
        if not isinstance(result, np.ndarray) or result.ndim == 0:
            return result
        axes = range(self.ndim) if axis is None else np.atleast_1d(axis) % self.ndim
        if self.ndim < 3 or any(ax < 3 for ax in axes):
            return result.view(np.ndarray)
        return Volume(result, **self.header)

    # reductions which are not ufuncs

    def argmax(self, axis=None, out=None, **kwargs):
        result = super().argmax(axis=axis, out=out, **kwargs)
        return result if out is not None else self._wrap_reduction(result, axis)

    def argmin(self, axis=None, out=None, **kwargs):
        result = super().argmin(axis=axis, out=out, **kwargs)
        return result if out is not None else self._wrap_reduction(result, axis)

    # axis permutations

    def transpose(self, *axes):
        result = super().transpose(*axes)
        if not axes or axes == (None,):
            order = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and not isinstance(axes[0], (int, np.integer)):
            order = tuple(axes[0])
        else:
            order = axes
        return self._permute_header(result, order)

    @property
    def T(self):
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        result = super().swapaxes(axis1, axis2)
        order = list(range(self.ndim))
        axis1, axis2 = axis1 % self.ndim, axis2 % self.ndim
        order[axis1], order[axis2] = order[axis2], order[axis1]
        return self._permute_header(result, order)

    def _permute_header(self, result, order):
        """ permute spacing and transform: axis i of result is axis order[i] of self """
        if self.ndim < 3:
            return result
        order = [ax % self.ndim for ax in order]
        if sorted(order[:3]) != [0, 1, 2]:
            # spatial axes moved beyond the first 3 axes
            return result.view(np.ndarray)
        result.spacing = [self.spacing[ax] for ax in order[:3]]
        result.transform = [self.transform[ax] for ax in order[:3]]
        return result

    # for pickle
    def __reduce__(self):
        """store info"""
        reduced = super().__reduce__()
        attrs = self._get_volume_attrs(self)
        reduced = reduced[:2] + (reduced[2] + (attrs,),)
        return reduced

    def __setstate__(self, state):
        """retrieve info"""
        attrs = state[-1]
        super().__setstate__(state[:-1])
        self._set_volume_attrs(self, attrs)

    @property
    def A(self) -> npt.NDArray[np.generic]:
        """return numpy.ndarray"""
        return self.view(np.ndarray)

    @property
    def spacing(self) -> tuple[float]:
        return self._spacing

    @spacing.setter
    def spacing(self, spacing: tuple[float] | None) -> None:
        if spacing is None:
            spacing = [1.0] * 3
        else:
            spacing = list(spacing)
            if len(spacing) != 3:
                raise ValueError
        self._spacing = tuple(float(v) for v in spacing)

    @property
    def origin(self) -> tuple[float]:
        return self._origin

    @origin.setter
    def origin(self, origin: tuple[float] | None) -> None:
        if origin is None:
            origin = [0.0] * 3
        else:
            origin = list(origin)
            if len(origin) != 3:
                raise ValueError
        self._origin = tuple(float(v) for v in origin)

    @property
    def transform(self) -> tuple[tuple[float]]:
        return self._orientation

    @transform.setter
    def transform(self, transform: tuple[tuple[float]] | None) -> None:
        if transform is None:
            transform = np.eye(3)
        else:
            if np.size(transform) != 9:
                raise ValueError
            transform = np.array(transform).reshape(3, 3)
        norm = np.linalg.norm(transform, axis=1, keepdims=True)
        norm[np.isclose(norm, 0)] = 1
        # store in column major order
        self._orientation = tuple(tuple(map(float, vec)) for vec in (transform / norm))

    @property
    def metadata(self) -> dict[str, str]:
        return self._metadata
    
    @metadata.setter
    def metadata(self, metadata: dict[str, str] | None) -> None:
        if metadata is None:
            metadata = {}
        self._metadata = {str(key): str(value) for key, value in metadata.items()}
    
    @property
    def affine(self):
        return make_affine_transform(self.origin, self.spacing, self.transform)
    
    @property
    def header(self):
        return self._get_volume_attrs(self)

    @property
    def anatomical_orientation(self) -> str:
        return get_orientation(self)
    
    def reorient(self, orientation: str):
        return set_orientation(self, orientation)

    # utilities
        
    def get_coords(self, indices):
        """compute real-world coordinates of a sequence of pixels"""
        is_flat = np.isscalar(indices[0])
        indices = np.atleast_2d(indices)
        coords = self.affine @ np.c_[indices, np.ones(len(indices))].T
        return coords[:3, 0] if is_flat else coords[:3].T

    def get_indices(self, coords):
        """compute real-world coordinates of a sequence of pixels"""
        is_flat = np.isscalar(coords[0])
        coords = np.atleast_2d(coords)
        indices = np.linalg.solve(self.affine, np.c_[coords, np.ones(len(coords))].T)
        return indices[:3, 0] if is_flat else indices[:3].T    
    



def get_spatial_slices(key, shape):
    """ start index and step of each spatial axis (0, 1, 2) for an indexing key

    returns None if the key is not made of slices on the spatial axes
    (or slices / integers on the other axes)
    """
    key = key if isinstance(key, tuple) else (key,)
    num_ellipsis = sum(item is Ellipsis for item in key)
    if num_ellipsis > 1:
        return None
    expanded = []
    for item in key:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (len(shape) - len(key) + 1))
        else:
            expanded.append(item)
    expanded.extend([slice(None)] * (len(shape) - len(expanded)))

    starts, steps = [], []
    for axis, item in enumerate(expanded):
        if isinstance(item, slice):
            start, _, step = item.indices(shape[axis])
            if axis < 3:
                starts.append(start)
                steps.append(step)
        elif axis >= 3 and isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            continue
        else:
            # integer on a spatial axis, array, mask, None...
            return None
    return starts, steps


# affine transform

def make_affine_transform(origin=None, spacing=None, rotation=None):
    """make transform matrix

    origin: a 3d vector
    spacing: a 3d vector
    rotation: a 3-sequence of 3d *columns*
        warning: when passing a 2d array, use the *transposed* rotation matrix

    """
    # source geometry
    if origin is None:
        origin = (0, 0, 0)
    if spacing is None:
        spacing = (1, 1, 1)
    if rotation is None:
        # assumes identify
        rotation = np.eye(3)

    origin = np.asarray(origin)
    spacing = np.asarray(spacing)

    # normalize rotation (assume it can be given as 1d array in column-major order)
    rotation = np.asarray(rotation).reshape(3, 3).T
    rotation = rotation / np.linalg.norm(rotation, axis=0, keepdims=True)

    # affine transform (rotation * scaling + translation)
    affine = np.r_[np.c_[rotation * spacing, origin], [[0, 0, 0, 1]]]

    return affine


def split_affine_transform(affine):
    """split affine transform into origin, spacing and rotation

    warning: rotation is the transposed of the rotation matrix
         ie. rotation can be seen as a sequence of *columns*
    """
    assert affine.shape == (4, 4)
    origin = affine[:-1, -1]
    spacing = np.linalg.norm(affine[:-1, :-1], axis=0)
    # transpose transform: return as sequence of columns
    rotation = np.transpose(1 / spacing * affine[:-1, :-1])

    return origin, spacing, rotation

    

#
# Orientation

ORIENTATIONS = [["R", "L"], ["A", "P"], ["I", "S"]]

def check_orientation(code):
    if len(code) != 3:
        raise ValueError(f"Invalid orientation code: {code}")
    if set(code.replace("L", "R").replace("P", "A").replace("S", "I")) != {
        "R",
        "A",
        "I",
    }:
        raise ValueError(f"Invalid orientation code: {code}")
    return True


def get_orientation(vol):
    """get volume anatomical orientation """

    transform = getattr(vol, "transform", np.eye(3))

    # assumes transform is a sequence of columns
    transform = np.asarray(transform).reshape(3, 3).T

    indices = np.argmax(np.abs(transform), axis=0)[:3]
    signs = 1 * (np.sign(transform[indices, np.arange(3)]) < 0)
    orient = "".join(ORIENTATIONS[i][s] for i, s in zip(indices, signs))

    check_orientation(orient)
    return orient


def set_orientation(vol, orientation, *, transform=None, copy=True):
    """set volume anatomical orientation"""

    check_orientation(orientation)

    vol = asvolume(vol, transform=transform)
    if copy:
        vol = vol.copy()
    source = get_orientation(vol)

    shape = vol.shape
    transform = vol.transform
    spacing = list(vol.spacing)
    center = [0, 0, 0]
    indices = [0, 1, 2]
    signs = [1, 1, 1]

    for i, code in enumerate(orientation):
        if code in source:
            index = source.find(code)
            sign = 1
        else:
            pair = [pair for pair in ORIENTATIONS if code in pair][0]
            alt = pair[1 - pair.index(code)]
            index = source.find(alt)
            sign = -1

        # invert direction and swap columns
        center[index] = (shape[index] - 1) if (sign < 0) else 0
        indices[i] = index
        signs[index] = sign

    # new origin and spacing
    origin = vol.get_coords(center)
    spacing = [spacing[i] for i in indices]

    # new transform
    transform = [[v * signs[i] for v in transform[i]] for i in indices]

    # flip and swap axes
    vol = np.flip(vol, axis=tuple(i for i in range(3) if signs[i] < 0))
    vol = np.moveaxis(vol, indices, [0, 1, 2])

    # set new values
    vol.origin = origin[:3]
    vol.transform = transform
    vol.spacing = spacing
    return vol    
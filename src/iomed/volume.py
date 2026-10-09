
from __future__ import annotations

import pathlib
import dataclasses
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

    filename = pathlib.Path(filename)
    if not filename.suffix:
        filename = filename.with_suffix(DEFAULT_EXT)

    data, options = _to_image_data(vol)
    imagefile.write_image(filename, data, use_compression=use_compression, compression_level=compression_level, **options)


def read(filename, *, nan_as=None, as_complex=False, dtype=None):
    """ read Volume

    as_complex: combine a last axis of size 2 (real, imaginary) into complex values
    dtype: cast data to dtype
    """
    array, header = imagefile.read_image(filename)
    return _from_image_data(array, header, nan_as=nan_as, as_complex=as_complex, dtype=dtype)


def _to_image_data(vol):
    """ data and options for imagefile.to_image / write_image """
    # complex: vector image with 2 components (real, imaginary)
    is_complex = np.iscomplexobj(vol)
    data = np.stack([vol.real, vol.imag], axis=-1) if is_complex else vol
    options = dict(spacing=vol.spacing, origin=vol.origin, transform=vol.transform, metadata=vol.metadata, is_vector=is_complex)
    return data, options


def _from_image_data(array, header, *, nan_as=None, as_complex=False, dtype=None):
    """ Volume from the outputs of imagefile.from_image / read_image """
    if nan_as is not None:
        array[array==nan_as] = np.nan
    if as_complex and array.ndim > 3 and array.shape[-1] == 2:
        array = array[..., 0] + 1j * array[..., 1]
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return Volume(array, **header)


# interoperability

# LPS (ITK, iomed) <-> RAS (NIfTI, nibabel) world coordinates
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def to_sitk(vol):
    """ convert Volume to SimpleITK Image (complex: vector image with 2 components) """
    data, options = _to_image_data(asvolume(vol))
    return imagefile.to_image(data, **options)


def from_sitk(image, *, as_complex=False, dtype=None) -> Volume:
    """ convert SimpleITK Image to Volume (vector images: components in the last axis)

    as_complex: combine a last axis of size 2 (real, imaginary) into complex values
    """
    array, header = imagefile.from_image(image)
    return _from_image_data(array, header, as_complex=as_complex, dtype=dtype)


def to_nibabel(vol):
    """ convert Volume to nibabel Nifti1Image (requires nibabel) """
    import nibabel  # lazy import: optional dependency

    vol = asvolume(vol)
    data = vol.A.astype(np.uint8) if vol.dtype == bool else vol.A
    image = nibabel.Nifti1Image(data, LPS_TO_RAS @ vol.affine)
    image.header.set_xyzt_units("mm")
    return image


def from_nibabel(image) -> Volume:
    """ convert nibabel image (with an affine, e.g. Nifti1Image) to Volume """
    origin, spacing, transform = split_affine_transform(LPS_TO_RAS @ image.affine)
    return Volume(np.asanyarray(image.dataobj), spacing=spacing, origin=origin, transform=transform)


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


@dataclasses.dataclass(frozen=True)
class Geometry:
    """ geometry of a 3d volume (or of the first 3 axes of a N-d volume)

    shape: spatial shape (x, y, z)
    spacing: voxel size
    origin: world coordinates of voxel (0, 0, 0)
    transform: world directions of the array axes (columns of the ITK direction matrix)
    """
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    transform: tuple[tuple[float, float, float], ...] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def __post_init__(self):
        # validate and normalize (same rules as Volume)
        shape = tuple(int(n) for n in self.shape)
        if len(shape) != 3:
            raise ValueError(f"Invalid shape (3 values expected): {self.shape}")
        values = dict(shape=shape, spacing=_check_vector(self.spacing, "spacing"), origin=_check_vector(self.origin, "origin"))
        values["transform"] = _check_transform(self.transform)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_dict(cls, dct):
        """ geometry from a dict with keys: shape, [spacing, origin, transform] """
        return cls(**{key: dct[key] for key in ("shape", "spacing", "origin", "transform") if key in dct})

    def to_dict(self):
        return dataclasses.asdict(self)

    @property
    def affine(self):
        return make_affine_transform(self.origin, self.spacing, self.transform)

    @property
    def extent(self) -> tuple[float, float, float]:
        """ size covered by the voxels along each axis (shape * spacing) """
        return tuple(float(n * s) for n, s in zip(self.shape, self.spacing))

    @property
    def center(self) -> npt.NDArray[np.floating]:
        """ world coordinates of the volume center """
        return self.get_coords((np.array(self.shape) - 1) / 2)

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.spacing))

    def get_coords(self, indices):
        """compute real-world coordinates of a sequence of pixels"""
        return get_coords(self.affine, indices)

    def get_indices(self, coords):
        """compute pixel indices of a sequence of real-world coordinates"""
        return get_indices(self.affine, coords)

    def same_as(self, other, tol=1e-4) -> bool:
        """ same shape, and same spacing, origin and transform up to `tol` """
        other = as_geometry(other)
        return self.shape == other.shape and all(
            np.allclose(getattr(self, name), getattr(other, name), rtol=0, atol=tol)
            for name in ("spacing", "origin", "transform")
        )


def as_geometry(obj) -> Geometry:
    """ Geometry from a Geometry, a Volume or a dict """
    if isinstance(obj, Geometry):
        return obj
    elif isinstance(obj, dict):
        return Geometry.from_dict(obj)
    return asvolume(obj).geometry


def _check_vector(values, name):
    values = list(values)
    if len(values) != 3:
        raise ValueError(f"Invalid {name} (3 values expected): {values}")
    return tuple(float(v) for v in values)


def _check_transform(transform):
    """ normalized transform: tuple of 3 axis directions """
    if np.size(transform) != 9:
        raise ValueError(f"Invalid transform (9 values expected): {transform}")
    transform = np.array(transform, dtype=float).reshape(3, 3)
    norm = np.linalg.norm(transform, axis=1, keepdims=True)
    norm[np.isclose(norm, 0)] = 1
    return tuple(tuple(map(float, vec)) for vec in (transform / norm))


def get_coords(affine, indices):
    """compute real-world coordinates of a sequence of pixels"""
    is_flat = np.isscalar(indices[0])
    indices = np.atleast_2d(indices)
    coords = affine @ np.c_[indices, np.ones(len(indices))].T
    return coords[:3, 0] if is_flat else coords[:3].T


def get_indices(affine, coords):
    """compute pixel indices of a sequence of real-world coordinates"""
    is_flat = np.isscalar(coords[0])
    coords = np.atleast_2d(coords)
    indices = np.linalg.solve(affine, np.c_[coords, np.ones(len(coords))].T)
    return indices[:3, 0] if is_flat else indices[:3].T


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
        geometry: Geometry | None = None,
        **kwargs,
    ) -> Volume:

        array = np.asanyarray(data, **kwargs).view(cls)
        if geometry is not None:
            # explicit spacing/origin/transform override the geometry
            geometry = as_geometry(geometry)
            if array.shape[:3] != geometry.shape:
                raise ValueError(f"Shape mismatch: data {array.shape}, geometry {geometry.shape}")
            array.spacing, array.origin, array.transform = geometry.spacing, geometry.origin, geometry.transform
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
        self._spacing = _check_vector([1.0] * 3 if spacing is None else spacing, "spacing")

    @property
    def origin(self) -> tuple[float]:
        return self._origin

    @origin.setter
    def origin(self, origin: tuple[float] | None) -> None:
        self._origin = _check_vector([0.0] * 3 if origin is None else origin, "origin")

    @property
    def transform(self) -> tuple[tuple[float]]:
        return self._orientation

    @transform.setter
    def transform(self, transform: tuple[tuple[float]] | None) -> None:
        # store as a sequence of columns (axis directions)
        self._orientation = _check_transform(np.eye(3) if transform is None else transform)

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
    def geometry(self) -> Geometry:
        """ geometry of the first 3 axes """
        return Geometry(shape=self.shape[:3], spacing=self.spacing, origin=self.origin, transform=self.transform)

    @property
    def anatomical_orientation(self) -> str:
        """ anatomical orientation code, ITK "from" convention (e.g. RAI) """
        return get_orientation(self)

    def get_orientation(self, convention: str = "from") -> str:
        """ anatomical orientation code

        convention: "from" (ITK, identity: RAI) or "to" (identity: LPS)
        """
        return get_orientation(self, convention=convention)

    # interoperability

    def to_sitk(self):
        """ convert to SimpleITK Image """
        return to_sitk(self)

    @classmethod
    def from_sitk(cls, image, **kwargs) -> Volume:
        """ convert SimpleITK Image to Volume (see `volume.from_sitk`) """
        return from_sitk(image, **kwargs)

    def to_nibabel(self):
        """ convert to nibabel Nifti1Image (requires nibabel) """
        return to_nibabel(self)

    @classmethod
    def from_nibabel(cls, image) -> Volume:
        """ convert nibabel image to Volume """
        return from_nibabel(image)

    def reorient(self, orientation: str, convention: str = "from"):
        """ flip/permute axes to match orientation code (see `get_orientation`) """
        return set_orientation(self, orientation, convention=convention)

    # utilities
        
    def get_coords(self, indices):
        """compute real-world coordinates of a sequence of pixels"""
        return get_coords(self.affine, indices)

    def get_indices(self, coords):
        """compute pixel indices of a sequence of real-world coordinates"""
        return get_indices(self.affine, coords)
    



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


def convert_orientation(code, convention):
    """ convert orientation code from `convention` ("from" or "to") to the "from" convention (or back) """
    if convention == "from":
        return code
    elif convention == "to":
        # opposite direction on each axis
        opposite = {a: b for pair in ORIENTATIONS for a, b in (pair, pair[::-1])}
        return "".join(opposite[c] for c in code)
    raise ValueError(f'Invalid orientation convention (expected "from" or "to"): {convention}')


def get_orientation(vol, *, convention="from"):
    """get volume anatomical orientation

    convention: "from" (ITK, identity: RAI) or "to" (identity: LPS)
    """

    transform = getattr(vol, "transform", np.eye(3))

    # assumes transform is a sequence of columns
    transform = np.asarray(transform).reshape(3, 3).T

    indices = np.argmax(np.abs(transform), axis=0)[:3]
    signs = 1 * (np.sign(transform[indices, np.arange(3)]) < 0)
    orient = "".join(ORIENTATIONS[i][s] for i, s in zip(indices, signs))

    check_orientation(orient)
    return convert_orientation(orient, convention)


def set_orientation(vol, orientation, *, transform=None, copy=True, convention="from"):
    """set volume anatomical orientation

    convention: "from" (ITK, identity: RAI) or "to" (identity: LPS)
    """

    check_orientation(orientation)
    orientation = convert_orientation(orientation, convention)

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
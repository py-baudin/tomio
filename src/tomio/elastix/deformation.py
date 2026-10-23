"""deformation analysis: spatial Jacobian, folding check, displacement field, SimpleITK export"""

from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from .api import _output, _output_geometry
from .transforms import TransformBSpline, TransformationSequence, as_transformation


def spatial_jacobian(transformation, points=None):
    """spatial Jacobian dT/dx of a transformation at points (npoint x ndim x ndim, world axes): analytic (B-spline
    weight derivatives, chain rule for sequences)

    points: world points (default: the points of the transformation's grid)
    """
    return as_transformation(transformation).jacobian(points)


def to_sitk(transformation):
    """SimpleITK transform equivalent to a transformation (same mapping of world points, fixed -> moving)

    global transforms: `sitk.AffineTransform`; B-splines: `sitk.BSplineTransform` (same control grid, padded;
    displacements differ outside the domain, where `TransformBSpline` clips); sequences: `sitk.CompositeTransform`
    """
    transformation = as_transformation(transformation)
    if isinstance(transformation, TransformationSequence):
        # ITK composite transforms apply the last added transform first
        return sitk.CompositeTransform([to_sitk(trans) for trans in transformation][::-1])
    ndim = transformation.geometry.ndim
    if transformation.is_global:
        A, t = transformation.affine()
        return sitk.AffineTransform([float(v) for v in A.ravel()], [float(v) for v in t])
    transform = transformation.transform
    if not isinstance(transform, TransformBSpline):
        raise TypeError(f"No SimpleITK equivalent for {transform}")
    grid = transform.grid
    shape = tuple(int(n) for n in grid["shape"])
    coeffs = np.reshape(transformation.parameters, (len(transform.components), -1))
    world = transform.basis @ coeffs  # world displacement coefficients (ndim x ncontrol)
    # one more control point on each side (edge values): same spline on the domain, which stays strictly inside
    # ITK's valid region (half-open, the borders of the domain would be lost to rounding)
    origin = grid["origin"] - grid["direction"] @ grid["spacing"]
    images = []
    for i in range(ndim):
        padded = np.pad(world[i].reshape(shape), 1, mode="edge")
        image = sitk.GetImageFromArray(np.ascontiguousarray(padded.T))
        image.SetOrigin([float(v) for v in origin])
        image.SetSpacing([float(v) for v in grid["spacing"]])
        image.SetDirection([float(v) for v in np.ravel(grid["direction"])])
        images.append(image)
    return sitk.BSplineTransform(images, 3)


def jacobian_determinant(transformation, ref=None, chunk=2**18):
    """determinant of the spatial Jacobian of a transformation on its grid (or on `ref`): local volume ratio
    (moving / fixed); <= 0 where the deformation folds (not invertible), 1 where it preserves volumes

    returns: tomio Volume (3d) or Image (2d)
    """
    transformation, geometry = _output_geometry(transformation, ref)
    points = geometry.points
    det = np.empty(len(points))
    for start in range(0, len(points), chunk):
        index = slice(start, start + chunk)
        det[index] = np.linalg.det(spatial_jacobian(transformation, points[index]))
    return _output(det.reshape(tuple(geometry.shape)), geometry, label="jacobian")


def check_folding(transformation, ref=None, mask=None):
    """folding check of a transformation: statistics of the Jacobian determinant on its grid (or on `ref`)

    mask: boolean array on the grid, points to check (default: all)

    returns: namespace with `determinant` (Volume or Image), `min`, `max`, `folded` (number of points with a
        determinant <= 0), `fraction` (folded fraction) and `ok` (no folding)
    """
    determinant = jacobian_determinant(transformation, ref=ref)
    values = np.asarray(determinant)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    folded = int(np.count_nonzero(values <= 0))
    return SimpleNamespace(
        determinant=determinant,
        min=float(values.min()) if values.size else np.nan,
        max=float(values.max()) if values.size else np.nan,
        folded=folded,
        fraction=folded / max(values.size, 1),
        ok=folded == 0,
    )


def deformation_field(transformation, ref=None):
    """displacement T(x) - x at the points of the transformation's grid (or of `ref`), world coordinates (mm)

    returns: array (grid shape + (ndim,))
    """
    transformation, geometry = _output_geometry(transformation, ref)
    points = geometry.points
    return (transformation(points) - points).reshape(tuple(geometry.shape) + (geometry.ndim,))

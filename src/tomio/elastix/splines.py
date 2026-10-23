"""spline interpolation, filtering and resampling of arrays (numpy, SimpleITK)"""

import math

import numpy as np
import SimpleITK as sitk

from .. import interp as _interp


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


def erode_mask(mask, radius):
    """erosion of a boolean mask as Elastix's ErodeMaskImageFilter (ITK parabolic erosion, scale r^2 / 2 + 1
    voxels^2 per axis): a voxel x is removed when a voxel y outside the mask has
    sum_i (x_i - y_i)^2 / (2 scale_i) < 1

    radius: per axis (voxels)
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.all() or not mask.any():
        return mask
    radius = np.broadcast_to(np.asarray(radius, dtype=float), (mask.ndim,))
    scale = radius**2 / 2 + 1
    outside = sitk.GetImageFromArray(np.ascontiguousarray((~mask).astype(np.uint8).T))
    outside.SetSpacing([float(v) for v in 1 / np.sqrt(2 * scale)])
    # distance (weighted) from each voxel of the mask to the nearest voxel outside
    distance = sitk.SignedMaurerDistanceMap(outside, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    return mask & (sitk.GetArrayFromImage(distance).T >= 1 - 1e-9)


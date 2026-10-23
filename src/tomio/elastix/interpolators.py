"""interpolators (image values and gradients at points)"""

from abc import ABC, abstractmethod

import numpy as np

from .image import Image, splitcomplex
from .splines import ndspline_interpolation


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

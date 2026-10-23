"""samplers (sample points of the fixed image)"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from .image import Image
from .splines import ndspline_interpolation

LOGGER = logging.getLogger("tomio.elastix")


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

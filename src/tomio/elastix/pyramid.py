"""multi-resolution pyramids"""

from abc import ABC, abstractmethod

import numpy as np

from .image import Image
from .splines import binary_dilation, gaussian_filter, resample_grid


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

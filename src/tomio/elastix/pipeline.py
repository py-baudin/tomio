"""registration pipeline: objective function, multi-resolution optimization"""

import logging

import numpy as np

from .image import Image
from .interpolators import Interpolator, InterpolatorNearestNeighbor
from .metrics import Metric
from .optimizers import OptimResults
from .samplers import Sampler
from .transforms import Transform, Transformation

LOGGER = logging.getLogger("tomio.elastix")


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
        limits = getattr(metric, "limits", None)
        self.normalize = lambda f, m: (fixed.normalize(f, limits=limits), moving.normalize(m, limits=limits))

    def moved_samples(self, samples):
        """sample points through the initial transformation (cached per sample set)"""
        if self.initial is None:
            return samples
        cached = getattr(self, "_moved", (None, None))
        if cached[0] is not samples:
            self._moved = (samples, self.initial(samples))
        return self._moved[1]

    def resample(self):
        """draw new samples (stochastic optimizers)"""
        if getattr(self.sampler, "image", None) is not None:
            self.sampler.update()

    def overlap(self, samples, points1, parameters):
        """samples of the current set that are used: those mapped inside the moving image (pixel centers range) by
        the parameters of the first evaluation (start of the optimization), cached per sample set

        samples: sample points (identify the set), points1: after the initial transform"""
        if getattr(self, "_reference", None) is None:
            self._reference = np.array(parameters, dtype=float)
        cached = getattr(self, "_overlap", (None, None))
        if cached[0] is not samples:
            points2 = self.transform(self._reference, points1, center=self.center)
            self._overlap = (samples, self.moving.geometry.contains(points2, margin=0))
        return self._overlap[1]

    def padded(self, inset, points2, values2, grad2=None):
        """values of the samples of the current set (`inset`, see `overlap`): padding value (moving minimum) and
        zero gradient for the samples that left the moving image

        The sample set is defined at the start of the optimization: otherwise the transform could improve the cost by
        moving badly aligned samples out of the moving image.
        returns: values, samples used [, gradients]
        """
        outside = ~self.moving.geometry.contains(points2)
        values2 = np.where(outside, self.moving._min, values2) if np.any(outside) else values2
        used = inset & np.isfinite(values2)
        if self._moving_mask is not None:
            used &= outside | (InterpolatorNearestNeighbor()(self._moving_mask, points2) > 0)
        if grad2 is None:
            return values2, used
        if np.any(outside):
            grad2 = np.array(grad2)
            npoint = len(points2)
            grad2[:npoint][outside] = 0
            if len(grad2) > npoint:  # complex: imaginary part
                grad2[npoint:][outside] = 0
        return values2, used, grad2

    def valid(self, points2, values2):
        """samples inside the moving image (and its mask), with finite values"""
        valid = self.moving.geometry.contains(points2) & np.isfinite(values2)
        if self._moving_mask is not None:
            valid &= InterpolatorNearestNeighbor()(self._moving_mask, points2) > 0
        return valid

    def _edge_scale(self, image):
        """mean gradient magnitude of an image (for gradient-based metrics)"""
        magnitude = np.sqrt(sum(np.asarray(component.array) ** 2 for component in image.gradient_images()))
        values = magnitude[image.mask] if image.mask is not None else magnitude
        return float(np.mean(values)) or 1.0

    def _call_gradients(self, parameters, gradient=False, **kwargs):
        """objective with a metric of multi-channel images derived from the images (NGF: gradients, MIND:
        descriptors), given by `metric.channel_images(image, objective)`"""
        transform_args = {"center": self.center}
        points1, _ = self.sampler(self.fixed)
        points1_moved = self.moved_samples(points1)
        if getattr(self, "_fixed_samples", (None,))[0] is not points1:
            # fixed channels at the samples (unchanged until the samples change)
            channels = self.metric.channel_images(self.fixed, self)
            values1 = np.stack([self.interpolator(component, points1) for component in channels], axis=1)
            eta = getattr(self.metric, "eta", 1.0)
            eps = (eta * self._edge_scale(self.fixed), eta * self._edge_scale(self.moving))
            self._fixed_samples = (points1, values1, eps)
        _, values1, eps = self._fixed_samples
        metric_args = {"eps": eps, **kwargs}
        penalty = getattr(self.transform, "penalty", None)
        components = self.metric.channel_images(self.moving, self)

        # samples outside the moving image are kept with zero channels (e.g. no alignment for NGF): discarding them would
        # let the transform improve the cost by pushing badly aligned samples out of the image
        if not gradient:
            points2 = self.transform(parameters, points1_moved, **transform_args)
            values2 = np.stack([self.interpolator(component, points2) for component in components], axis=1)
            values2[~self.valid(points2, np.sum(values2, axis=1))] = 0
            cost = self.metric(values1, values2, **metric_args)
            return cost + penalty(parameters) if penalty is not None else cost

        points2, jtrans = self.transform(parameters, points1_moved, jac=True, **transform_args)
        interpolated = [self.interpolator(component, points2, jac="points") for component in components]
        values2 = np.stack([values for values, _ in interpolated], axis=1)
        dvalues2 = np.stack([grad for _, grad in interpolated], axis=1)  # npoint x component x ndim
        isvalid = self.valid(points2, np.sum(values2, axis=1))
        values2[~isvalid] = 0
        cost, gmetric = self.metric(values1, values2, gradient2=True, **metric_args)
        gmetric = gmetric[isvalid]
        gpoints = np.einsum("nc,ncd->nd", gmetric, dvalues2[isvalid])
        if isinstance(jtrans, np.ndarray):
            gradient = np.einsum("nd,ndk->k", gpoints, jtrans[isvalid])
        else:
            gradient = jtrans.rmatvec(gpoints, isvalid)
        if penalty is not None:
            pcost, pgrad = penalty(parameters, gradient=True)
            cost, gradient = cost + pcost, gradient + pgrad
        return cost, gradient

    def __call__(self, parameters, gradient=False, **kwargs):
        if getattr(self.metric, "uses_gradients", False):
            return self._call_gradients(parameters, gradient=gradient, **kwargs)
        transform_args = {"center": self.center}
        metric_args = {"normalize": self.normalize, **kwargs}
        points1, values1 = self.sampler(self.fixed)
        samples = points1  # identifies the sample set
        points1 = self.moved_samples(samples)
        penalty = getattr(self.transform, "penalty", None)
        if not gradient:
            points2 = self.transform(parameters, points1, **transform_args)
            values2 = self.interpolator(self.moving, points2)
            values2, isvalid = self.padded(self.overlap(samples, points1, parameters), points2, values2)
            cost = self.metric(values1[isvalid], values2[isvalid], **metric_args)
            if penalty is not None:
                cost = cost + penalty(parameters)
            LOGGER.debug(f"cost: {cost:.4}")
            return cost

        # else, also return gradient
        points2, jtrans = self.transform(
            parameters, points1, jac=True, **transform_args
        )
        values2, grad2 = self.interpolator(self.moving, points2, jac="points")
        values2, isvalid, grad2 = self.padded(self.overlap(samples, points1, parameters), points2, values2, grad2)
        cost, gmetric = self.metric(
            values1[isvalid], values2[isvalid], gradient2=True, **metric_args
        )
        # gradient of the cost w/r to the (valid) moving points
        if np.iscomplexobj(values2):
            npoint, nvalid = len(values2), np.count_nonzero(isvalid)
            gpoints = gmetric[:nvalid, None] * grad2[:npoint][isvalid] + gmetric[nvalid:, None] * grad2[npoint:][isvalid]
        else:
            gpoints = gmetric[:, None] * grad2[isvalid]
        # chain rule through the transform
        if isinstance(jtrans, np.ndarray):
            gradient = np.einsum("nd,ndk->k", gpoints, jtrans[isvalid])
        else:
            gradient = jtrans.rmatvec(gpoints, isvalid)
        if penalty is not None:
            pcost, pgrad = penalty(parameters, gradient=True)
            cost, gradient = cost + pcost, gradient + pgrad
        LOGGER.debug(f"cost: {cost:.4}")
        return cost, gradient


class ScaledObjective:
    """objective function of scaled parameters: x = parameters * scales"""

    def __init__(self, objective, scales):
        self.objective = objective
        self.scales = np.asarray(scales, dtype=float)

    def __call__(self, x, gradient=False, **kwargs):
        parameters = np.asarray(x) / self.scales
        if not gradient:
            return self.objective(parameters, **kwargs)
        cost, grad = self.objective(parameters, gradient=True, **kwargs)
        return cost, grad / self.scales

    def resample(self):
        self.objective.resample()


class Elastix:
    """Main registration object"""

    def __init__(self, config, loglevel=None):
        self.config = config
        self.logger = LOGGER
        loglevel = loglevel if loglevel else "WARN"  # default log-level
        self.logger.setLevel(getattr(logging, str(loglevel), loglevel))

    def execute(self, fixed: Image, moving: Image, init=None, callback=None, initial=None):
        """compute transform

        init: initial parameters
        callback: called after each level: callback(fixed, moving, parameters, config, level=level)
        initial: transformation applied before the optimized transform (previous stages)
        """
        LOGGER.info("Execute pipeline")

        pyramid = self.config.setup("pyramid")
        if getattr(pyramid, "downsample", False) is None:
            # downsampling; for non-global transforms, not below a minimum size (tiny levels can diverge)
            pyramid.downsample = True
            if not self.config.setup("transform").is_global:
                pyramid.minsize = max(pyramid.minsize, pyramid.MINSIZE_DEFORMABLE)
        optimizer = self.config.setup("optimizer")
        self.logger.info(f"Using pyramid: {pyramid}")
        self.logger.info(f"Using optimizer: {optimizer}")

        # compute pyramid
        levels = pyramid(fixed, moving)

        # same center of rotation at all levels
        center = fixed.geometry.center
        radius = fixed.geometry.radius(center)
        parameters = None if init is None else np.asarray(init, dtype=float)
        method = self.config["initialize"].get("method")
        if parameters is None and initial is None and method is not None:
            transform = self.config.setup("transform")
            transform.setup(fixed.geometry)
            parameters = initialize(transform, fixed, moving, method)
        results = []
        previous = None  # transform of the previous level
        for level, images in zip(*levels):
            _fixed, _moving = images

            self.logger.info(f"Optimize level: {level}")
            self.logger.debug(f"Shape: fixed={_fixed.shape}, moving={_moving.shape}")
            self.logger.debug(
                f"Spacing: fixed={_fixed.spacing}, moving={_moving.spacing}"
            )

            transform = self.config.setup("transform")
            # full resolution fixed image (e.g. same center at all levels); level relative to the finest one
            transform.setup(fixed.geometry, level=level - levels[0][-1])
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
            if parameters is None:
                init = transform.init(fixed.ndim)
            elif previous is not None:
                init = transform.refine(previous, parameters)
            else:
                init = parameters
            previous = transform
            self.logger.debug(f"Initialization: {init}")

            # optimize scaled parameters (one unit ~ 1 mm displacement)
            scales = transform.scales(init, radius)
            scaled = ScaledObjective(objective, scales)
            parameters, info = optimizer(scaled, np.asarray(init, dtype=float) * scales)
            parameters = parameters / scales

            nsample = len(sampler(_fixed)[0])
            self.logger.debug(f"num samples: {nsample}")
            self.logger.debug(info)
            self.logger.info(f"Parameters: {parameters}")
            results.append(
                OptimResults(
                    level=level, fixed_shape=_fixed.shape, moving_shape=_moving.shape, nsample=nsample,
                    parameters=parameters, **{key: getattr(info, key, None) for key in ("fun", "nit", "nfev", "message", "success")}
                )
            )

            if callback:
                callback(_fixed, _moving, parameters, self.config, level=level)

        transformation = Transformation(transform, fixed.geometry, parameters, center=center)
        self.logger.info(f"Found transformation: {transformation}")
        self.logger.info(f"Done.")
        info.levels = results
        return transformation, info


def initialize(transform, fixed, moving, method):
    """initial parameters: translation aligning the image centers ("geometry") or centers of mass ("moments")"""
    if method == "geometry":
        offset = moving.geometry.center - fixed.geometry.center
    elif method == "moments":
        offset = center_of_mass(moving) - center_of_mass(fixed)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    parameters = transform.with_translation(transform.init(fixed.ndim), offset)
    if parameters is None:
        LOGGER.warning(f"Initialization '{method}' is not supported by {transform}: ignored")
    return parameters


def center_of_mass(image):
    """intensity-weighted center (of the masked values), world coordinates"""
    weights = np.abs(image.values)
    if not np.any(weights):
        return image.geometry.center
    return np.average(image.points, axis=0, weights=weights)

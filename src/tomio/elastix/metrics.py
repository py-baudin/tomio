"""similarity metrics"""

from abc import ABC, abstractmethod

import numpy as np

from .image import Image, splitcomplex
from .splines import gaussian_filter


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
            return np.inf if not gradient2 else (np.inf, np.zeros(values2.shape))
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


class MetricNormalizedGradientFields(Metric):
    """Normalized gradient fields (Haber & Modersitzki 2006): 1 - mean (n_f . n_m)^2, n = grad / sqrt(|grad|^2 + eps^2)

    Multimodal (compares edge directions, independent of the contrast), well suited to free-form deformations.
    eta: edge parameter, eps = eta x mean gradient magnitude of each image (small eta: less bias towards
        transforms increasing the moving gradients, more sensitivity to noise)
    samples mapped outside the moving image count as unaligned (not discarded)
    """

    uses_gradients = True

    def __init__(self, eta=0.01):
        self.eta = eta

    def channel_images(self, image, objective=None):
        """gradient images (world axes)"""
        return image.gradient_images()

    def measure(self, values1, values2, gradient2=False, eps=(1.0, 1.0), **kwargs):
        """values1, values2: gradients of the fixed and moving images at the samples (nsample x ndim)"""
        r1 = np.sqrt(np.sum(values1**2, axis=1) + eps[0] ** 2)
        r2 = np.sqrt(np.sum(values2**2, axis=1) + eps[1] ** 2)
        n1 = values1 / r1[:, np.newaxis]
        dot = np.sum(n1 * values2, axis=1) / r2
        cost = 1 - np.mean(dot**2)
        if not gradient2:
            return cost
        # d(dot) / d(values2) = n1 / r2 - (n1 . values2) values2 / r2^3
        ddot = n1 / r2[:, np.newaxis] - (dot / r2**2)[:, np.newaxis] * values2
        return cost, -2 / len(dot) * dot[:, np.newaxis] * ddot


class MetricMIND(Metric):
    """Modality independent neighbourhood descriptor (Heinrich et al., Medical Image Analysis 16(7), 2012)

    For each image and each offset r = +/-1 pixel along the (non-thick) axes: patch distance
    D_r = G_sigma * (I - I(. + r))^2, local variance V = mean_r D_r, descriptor MIND_r = exp(-D_r / V) normalized
    to a maximum of 1 over r. Metric: mean squared difference of the descriptors (structure, not intensities:
    for images of different contrasts). Axes thick in either image (spacing >= thick_ratio x the smallest
    spacing) are excluded, so that both images have the same channels.

    sigma: gaussian patch size (pixels)
    """

    uses_gradients = True  # multi-channel images (objective path of NGF)

    def __init__(self, sigma=1.0, thick_ratio=2.0):
        self.sigma = sigma
        self.thick_ratio = thick_ratio

    def axes(self, *images):
        """axes used for the offsets: not thick in any image"""
        thick = np.zeros(images[0].ndim, dtype=bool)
        for image in images:
            spacing = np.asarray(image.spacing, dtype=float)
            thick |= spacing >= self.thick_ratio * spacing.min()
            thick |= np.array(image.shape) < 3
        return tuple(int(a) for a in np.flatnonzero(~thick))

    def channel_images(self, image, objective=None):
        axes = self.axes(objective.fixed, objective.moving) if objective is not None else self.axes(image)
        key = ("mind", axes, self.sigma)
        if key not in image._coefficients:
            descriptors = mind_descriptors(np.asarray(image.array), axes, self.sigma)
            image._coefficients[key] = [
                Image(channel, spacing=image.spacing, origin=image.origin, direction=image.direction) for channel in descriptors
            ]
        return image._coefficients[key]

    def measure(self, values1, values2, gradient2=False, **kwargs):
        """values1, values2: descriptors at the samples (nsample x channels)"""
        diff = values2 - values1
        cost = np.mean(diff**2)
        if not gradient2:
            return cost
        return cost, 2 * diff / diff.size


def mind_descriptors(array, axes, sigma=1.0):
    """MIND descriptors (one array per offset +/-1 along `axes`)"""
    array = np.abs(array) if np.iscomplexobj(array) else np.asarray(array, dtype=float)
    blur = np.zeros(array.ndim)
    blur[list(axes)] = sigma  # patches along the offset axes only
    distances = []
    for axis in axes:
        for step in (-1, 1):
            shifted = np.take(array, np.clip(np.arange(array.shape[axis]) + step, 0, array.shape[axis] - 1), axis=axis)
            distances.append(gaussian_filter((array - shifted) ** 2, blur))
    distances = np.stack(distances)
    variance = distances.mean(axis=0)
    mean = variance.mean()
    variance = np.clip(variance, 1e-3 * mean + 1e-12, 1e3 * mean + 1e-12)
    # exp(-D / V) normalized to a maximum of 1: exp(-(D - min D) / V) (no underflow)
    return np.exp(-(distances - distances.min(axis=0)) / variance)


class MetricMutualInformation(Metric):
    """Mutual information

    nbin: number of histogram bins
    limits: intensity range of each image (percentiles), e.g. (0.5, 99.5); values outside are clipped to the
        first / last bin; default (None): min and max (outliers can squeeze the values into a few bins)
    """

    def __init__(self, nbin=32, limits=None):
        self.nbin = nbin
        self.limits = None if limits is None else tuple(limits)

    def measure(self, values1, values2, gradient2=False, normalize=False, **kwargs):
        """mutual information (negative, to be minimized)

        normalize:
            False: values are already in [0, 1]
            True: normalize each set of values by its own range (gradient: range taken as constant)
            callable: normalize(values1, values2), affine in values2 (e.g. using the images' ranges)
        """
        opts = {"windows": ["spline0", "spline3"], "nbin": self.nbin, **kwargs}
        scale = 1.0
        if callable(normalize):
            # derivative of the (affine) normalization of values2
            ref = normalize(np.zeros(2), np.array([0.0, 1.0]))[1]
            scale = ref[1] - ref[0]
            values1, values2 = normalize(values1, values2)
        elif normalize:
            scale = 1 / (np.ptp(values2) + 1e-8)
            values1, values2 = normalize_values(values1), normalize_values(values2)
        # values out of the histogram range (e.g. spline overshoot): clipped, no gradient
        inside = (values2 >= 0) & (values2 <= 1)
        values1, values2 = np.clip(values1, 0, 1), np.clip(values2, 0, 1)
        if not gradient2:
            return mutualinfo(values1, values2, **opts)
        cost, grad = mutualinfo(values1, values2, gradient2=True, **opts)
        return cost, grad * scale * inside


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
    def spline0(x):
        """0-order spline window"""
        return 1.0 * ((x >= -0.5) & (x < 0.5))

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


def normalize_values(values, tol=1e-8):
    """normalize values between 0 and 1"""
    values = np.asarray(values)
    ptp = np.ptp(values)
    if np.isclose(ptp, 0):
        return np.zeros(values.shape)
    return (values - values.min()) / (ptp + tol)


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


def _log_ratio(p12):
    """log(p12 / (p1 p2)), 0 where p12 = 0"""
    p1p2 = np.outer(p12.sum(axis=1), p12.sum(axis=0))
    nonzero = (p12 > 0) & (p1p2 > 0)
    log12 = np.zeros_like(p12)
    log12[nonzero] = np.log(p12[nonzero] / p1p2[nonzero])
    return log12


def mutualinfo_scatter(values1, values2, nbin=32, gradient2=False):
    """mutual information (negative) with zero-order (fixed) and cubic (moving) Parzen windows, values in [0, 1]

    Same result as `mutualinfo(..., windows=["spline0", "spline3"])`, by scatter-add: each sample adds to
    1 x 4 bins of the joint histogram, O(N) instead of O(N nbin).
    """
    nvalue = len(values1)
    # fixed: bin of each value (window [-0.5, 0.5))
    bins1 = np.clip(np.ceil((nbin - 1) * values1 - 0.5), 0, nbin - 1).astype(np.intp)
    # moving: 4 cubic taps, folded into the edge bins
    u2 = (nbin - 1) * values2
    taps = np.floor(u2)[:, np.newaxis] + np.arange(-1, 3)
    dist = taps - u2[:, np.newaxis]
    bins2 = np.clip(taps, 0, nbin - 1).astype(np.intp)
    flat = bins1[:, np.newaxis] * nbin + bins2
    weights = Kernel.spline3(dist)
    p12 = np.bincount(flat.ravel(), weights=weights.ravel(), minlength=nbin * nbin).reshape(nbin, nbin) / nvalue
    log12 = _log_ratio(p12)
    mi = -(p12 * log12).sum()
    if not gradient2:
        return mi
    # d window / d value2 = -(nbin - 1) B3'(dist), B3'(x) = B2(x + 1/2) - B2(x - 1/2)
    dweights = Kernel.spline2(dist + 0.5) - Kernel.spline2(dist - 0.5)
    grad = (nbin - 1) / nvalue * (dweights * log12.ravel()[flat]).sum(axis=1)
    return mi, grad


def mutualinfo(values1, values2, gradient2=False, **kwargs):
    """mutual information metric"""
    windows = list(kwargs.get("windows", ["spline0", "spline3"]))
    if windows == ["spline0", "spline3"] and not kwargs.get("normalize"):
        return mutualinfo_scatter(values1, values2, nbin=kwargs.get("nbin", 10), gradient2=gradient2)

    # estimate joint distribution
    jac2 = "split" if gradient2 else False
    res = parzen2d(values1, values2, jac2=jac2, **kwargs)
    p12, jp12 = res if gradient2 else (res, None)

    # mutual information
    log12 = _log_ratio(p12)
    mi = -(p12 * log12).sum()

    if not gradient2:
        return mi

    # gradient w/r to 2nd argument (values2)
    # grad = - np.einsum("ijk,ij->k", jp12, log12)
    # merge jacobian parts and log12
    grad = -np.einsum("ik,jk,ij->k", *jp12, log12)
    return mi, grad

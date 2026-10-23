"""Image registration: an implementation of the Elastix registration framework

cf:
- https://en.wikipedia.org/wiki/Elastix_(image_registration)
- https://elastix.lumc.nl/doxygen/classelastix_1_1RandomSampler.html
- https://simpleelastix.readthedocs.io/ParameterMaps.html

usage:
    from tomio import elastix
    registered, trans = elastix.register(fixed, moving, transform="rigid", metric="mi", return_transform=True)
    roi_registered = elastix.transform(trans, roi, ref=fixed, method="roi")
    elastix.check_folding(trans).ok  # deformable transformations: invertible (Jacobian determinant > 0)

configuration: see `Config` and `Config.describe()`

modules: api (register, transform), config (sections, registry), image, pyramid, transforms, samplers,
interpolators, metrics, optimizers, pipeline (objective, multi-resolution optimization), splines (array
interpolation and filtering), deformation (Jacobian, folding, SimpleITK export), slices (sparse ROIs)
"""

import logging

from .splines import (  # noqa: F401
    bspline_coefficients,
    bspline_kernel,
    bspline_kernel_derivative,
    mirror_index,
    _spline_weights,
    _spline_evaluate,
    ndspline_interpolation,
    resample_grid,
    gaussian_filter,
    binary_dilation,
)
from .image import (  # noqa: F401
    direction_to_transform,
    transform_to_direction,
    _tolist,
    splitcomplex,
    Geometry,
    get_direction,
    Image,
)
from .pyramid import (  # noqa: F401
    Pyramid,
    PyramidDummy,
    PyramidGaussian,
)
from .transforms import (  # noqa: F401
    _finite_jacobian,
    Transformation,
    TransformationSequence,
    as_transformation,
    Transform,
    TransformTranslation,
    TransformRigid,
    TransformAffine,
    TransformBSpline,
    _second_difference,
    _second_difference_adjoint,
    SparseJacobian,
    normalaxes3d,
    translation,
    rigidtransform2d,
    rigidinverse2d,
    euler_matrices,
    euler_to_matrix,
    matrix_to_euler,
    rigidtransform3d,
    eulertransform3d,
    rigidinverse3d,
    affinetransform2d3d,
)
from .samplers import (  # noqa: F401
    Sampler,
    SamplerFull,
    SamplerRandom,
    SamplerRandomCoordinates,
)
from .interpolators import (  # noqa: F401
    Interpolator,
    InterpolatorNearestNeighbor,
    InterpolatorLinear,
    InterpolatorBSpline,
)
from .metrics import (  # noqa: F401
    Metric,
    MetricMeanSquares,
    MetricNormalizedGradientFields,
    MetricMIND,
    mind_descriptors,
    MetricMutualInformation,
    Kernel,
    normalize_values,
    parzen2d,
    _log_ratio,
    mutualinfo_scatter,
    mutualinfo,
)
from .optimizers import (  # noqa: F401
    Optimizer,
    OptimizerLBFGS,
    OptimizerSGD,
    OptimizerAuto,
    lbfgs,
    OptimResults,
)
from .config import (  # noqa: F401
    make_stages,
    COMPONENTS,
    get_component,
    component_name,
    component_options,
    Config,
)
from .pipeline import (  # noqa: F401
    Objective,
    ScaledObjective,
    Elastix,
    initialize,
    center_of_mass,
)
from .api import (  # noqa: F401
    register,
    transform,
    _apply,
    SITK_INTERPOLATORS,
    _resample_sitk,
    _output,
    _output_geometry,
    warp_labels,
    warp_roi,
)
from .deformation import (  # noqa: F401
    spatial_jacobian,
    to_sitk,
    jacobian_determinant,
    check_folding,
    deformation_field,
)
from .slices import (  # noqa: F401
    SLICE_REGISTRATION,
    interpolate_slices,
    _signed_distance,
    _warp_distances,
    _fuse_distances,
)

LOGGER = logging.getLogger("tomio.elastix")

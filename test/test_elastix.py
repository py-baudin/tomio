""" tests for tomio.elastix

Phase 0 (PLAN_elastix.md): characterization of the imported module.
Known bugs are `xfail(strict=True)` tests named after their id in the plan: fixing a bug flips its test.
"""
import numpy as np
import pytest

from tomio import elastix as ex


# helpers

def phantom(shape, spacing=None, *, inverse=None, seed=0, nblob=12, scale=1.0):
    """smooth image: sum of gaussian blobs, evaluated exactly at (optionally transformed) pixel positions

    inverse: function mapping moving points to fixed points (moving(y) = fixed(inverse(y)))
    """
    rng = np.random.default_rng(seed)
    shape = tuple(shape)
    spacing = np.ones(len(shape)) if spacing is None else np.asarray(spacing, dtype=float)
    extent = np.array(shape) * spacing
    centers = rng.uniform(0.2, 0.8, size=(nblob, len(shape))) * extent
    widths = rng.uniform(0.08, 0.15, size=(nblob, 1)) * extent.max()
    weights = rng.uniform(0.5, 1, size=nblob)
    points = np.reshape(np.indices(shape), (len(shape), -1)).T * spacing
    if inverse is not None:
        points = inverse(points)
    dist2 = (((points[:, np.newaxis] - centers) / widths) ** 2).sum(axis=-1)
    values = (weights * np.exp(-0.5 * dist2)).sum(axis=1).reshape(shape)
    return scale * values / values.max()


def shifted(shape, shift, **kwargs):
    """fixed and moving phantoms, moving(y) = fixed(y - shift): T(x) = x + shift"""
    fixed = phantom(shape, **kwargs)
    moving = phantom(shape, inverse=lambda y: y - np.asarray(shift), **kwargs)
    return fixed, moving


def rotated(shape, angle, shift, **kwargs):
    """fixed and moving phantoms related by a rotation (about the image center) and a translation"""
    center = (np.array(shape) - 1) / 2
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    if len(shape) == 3:
        rot = np.block([[rot, np.zeros((2, 1))], [np.zeros((1, 2)), np.ones((1, 1))]])

    def inverse(y):
        return (y - center - shift) @ rot + center

    return phantom(shape, **kwargs), phantom(shape, inverse=inverse, **kwargs)


def inner_error(registered, fixed, margin=6):
    """mean absolute difference away from the borders"""
    inner = tuple(slice(margin, -margin) if n > 2 * margin else slice(None) for n in fixed.shape)
    return np.abs(np.asarray(registered) - fixed)[inner].mean()


def make_objective(fixed, moving, config=None, spacing=None):
    config = ex.Config(config)
    return ex.Objective(
        ex.Image(fixed, spacing=spacing),
        ex.Image(moving, spacing=spacing),
        config.setup("Transform"),
        config.setup("Sampler"),
        config.setup("Metric"),
        config.setup("Interpolator"),
    )


def finite_gradient(func, params, step=1e-5):
    params = np.asarray(params, dtype=float)
    return np.array([(func(params + step * e) - func(params - step * e)) / (2 * step) for e in np.eye(len(params))])


def known_bug(bug_id, reason):
    return pytest.mark.xfail(strict=True, reason=f"{bug_id}: {reason}")


TRANSLATION = {"Transform": "TransformTranslation"}
MI = {"Metric": "MetricMutualInformation"}


# characterization: what works today

@pytest.mark.parametrize("shape, shift", [((48, 48), (2.5, -1.5)), ((32, 32, 16), (1.5, -1.0, 0.5))])
def test_translation_mse(shape, shift):
    fixed, moving = shifted(shape, shift)
    registered, trans = ex.register(fixed, moving, config=TRANSLATION, return_transform=True)
    assert isinstance(trans, ex.Transformation)
    assert np.allclose(trans.parameters, shift, atol=0.05)
    assert inner_error(registered, fixed) < 1e-2


@known_bug("B10", "the rotation center changes between pyramid levels: translation not recovered")
def test_rigid_mse_3d():
    fixed, moving = rotated((32, 32, 16), 0.05, (1.5, -1.0, 0.5))
    registered = ex.register(fixed, moving)
    assert inner_error(registered, fixed) < 0.1 * inner_error(moving, fixed)


def test_rigid_mi_2d():
    fixed, moving = rotated((48, 48), 0.05, (2.5, -1.5))
    registered = ex.register(fixed, moving, config=MI)
    assert inner_error(registered, fixed) < 0.2 * inner_error(moving, fixed)


def test_register_several_moving():
    fixed, moving = shifted((40, 40), (1.5, 1.0))
    registered, trans = ex.register(fixed, [moving, moving], config=TRANSLATION, return_transform=True)
    assert isinstance(registered, list) and len(registered) == 2
    assert isinstance(trans, list) and len(trans) == 2
    for params in trans:
        assert np.allclose(params.parameters, (1.5, 1.0), atol=0.05)


def test_register_complex():
    fixed, moving = shifted((40, 40), (1.5, 1.0))
    registered = ex.register(fixed * (1 + 1j), moving * (1 + 1j), config=TRANSLATION)
    assert np.iscomplexobj(np.asarray(registered))
    assert inner_error(np.real(registered), fixed) < 1e-2


def test_transform_matches_register():
    fixed, moving = shifted((40, 40), (1.5, 1.0))
    registered, trans = ex.register(fixed, moving, config=TRANSLATION, return_transform=True)
    assert np.allclose(np.asarray(ex.transform(trans, moving)), np.asarray(registered))
    # serialized transformation
    assert np.allclose(np.asarray(ex.transform(trans.serialize(), moving)), np.asarray(registered))


@pytest.mark.parametrize("transform", ["TransformTranslation", "TransformRigid", "TransformAffine"])
def test_serialize_roundtrip(transform):
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    _, trans = ex.register(fixed, moving, config={"Transform": transform}, return_transform=True)
    trans2 = ex.Transformation.deserialize(trans.serialize())
    assert type(trans2.transform) is type(trans.transform)
    assert np.allclose(trans2.parameters, trans.parameters)
    assert np.allclose(trans2(), trans())


@pytest.mark.parametrize(
    "config, params",
    [
        (TRANSLATION, [0.3, 0.2]),
        ({"Transform": "TransformRigid"}, [0.01, 0.3, 0.2]),
        ({"Transform": "TransformAffine"}, [1.01, 0.02, -0.01, 0.99, 0.3, 0.2]),
        ({**TRANSLATION, **MI}, [0.3, 0.2]),
    ],
)
def test_gradient_unit_spacing(config, params):
    fixed, moving = shifted((40, 40), (1.5, 1.0))
    objective = make_objective(fixed, moving, config)
    _, gradient = objective(params, gradient=True)
    assert np.allclose(gradient, finite_gradient(objective, params), rtol=1e-2, atol=1e-6)


def test_config_defaults():
    config = ex.Config()
    for component in ex.Config.COMPONENTS:
        assert config.setup(component) is not None
    with pytest.raises(ValueError):
        ex.Config({"Transform": None}).setup("Unknown")


# known bugs (PLAN_elastix.md, phase 1)

@known_bug("B1", "Elastix.execute ignores `init`")
def test_execute_uses_init():
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    config = ex.Config({**TRANSLATION, "Pyramid": "PyramidDummy", "OptimizerGeneric": {"options": {"maxiter": 0}}})
    trans, _ = ex.Elastix(config).execute(ex.Image(fixed), ex.Image(moving), init=[0.7, 0.3])
    assert np.allclose(trans.parameters, [0.7, 0.3])


@known_bug("B2", "transform(ref=...) uses `ref.geom`")
def test_transform_with_ref():
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    _, trans = ex.register(fixed, moving, config=TRANSLATION, return_transform=True)
    out = ex.transform(trans, moving, ref=ex.Image(np.zeros((20, 24))))
    assert np.asarray(out).shape == (20, 24)


@known_bug("B3", "serialize uses the bound method `self.init`")
@pytest.mark.parametrize("transform, init", [("TransformRigid", [0.1, 1.0, 2.0]), ("TransformAffine", [1, 0, 0, 1, 1.0, 2.0])])
def test_serialize_with_init(transform, init):
    trans = ex.Transformation(transform, ex.Geometry((8, 8)), init, init=np.asarray(init))
    data = ex.Transformation.deserialize(trans.serialize()).transform.serialize()
    assert np.allclose(data["Settings"]["init"], init)


@known_bug("B4", "TransformTranslation.serialize drops `axis`")
def test_serialize_translation_axis():
    transform = ex.Transform.deserialize(ex.TransformTranslation(axis=2).serialize())
    assert transform.axis == 2


@known_bug("B5", "linear interpolation gradient lacks the 1/spacing factor")
def test_gradient_anisotropic_spacing():
    spacing = (1.0, 1.0, 5.0)
    fixed, moving = shifted((24, 24, 12), (1.0, -1.0, 2.0), spacing=spacing)
    objective = make_objective(fixed, moving, TRANSLATION, spacing=spacing)
    params = [0.3, 0.2, 0.7]
    _, gradient = objective(params, gradient=True)
    assert np.allclose(gradient, finite_gradient(objective, params), rtol=1e-2, atol=1e-6)


@known_bug("B6", "MI gradient lacks the intensity normalization factor")
def test_gradient_mi_intensity_scale():
    fixed, moving = shifted((40, 40), (1.5, 1.0), scale=100)
    objective = make_objective(fixed, moving, {**TRANSLATION, **MI})
    params = [0.3, 0.2]
    _, gradient = objective(params, gradient=True)
    assert np.allclose(gradient, finite_gradient(objective, params), rtol=1e-2, atol=1e-6)


@known_bug("B7", "B-spline interpolator gradient has the wrong shape")
def test_register_bspline_interpolator():
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    _, trans = ex.register(fixed, moving, config={**TRANSLATION, "Interpolator": "InterpolatorBSpline"}, return_transform=True)
    assert np.allclose(trans.parameters, (1.0, 0.5), atol=0.05)


@known_bug("B8", "out-of-FOV moving samples count as 0")
def test_cost_out_of_fov():
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    objective = make_objective(fixed, moving, TRANSLATION)
    assert objective([1000.0, 0.0]) == np.inf  # no valid sample


@known_bug("B8", "out-of-FOV samples make the cost discontinuous at the image border (full sampling)")
def test_register_full_sampler():
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    _, trans = ex.register(fixed, moving, config={**TRANSLATION, "Sampler": "SamplerFull"}, return_transform=True)
    assert np.allclose(trans.parameters, (1.0, 0.5), atol=0.05)


@known_bug("B10", "Geometry.center is off by half a voxel")
def test_geometry_center():
    assert np.allclose(ex.Geometry((3, 5), spacing=(2, 1), origin=(10, 0)).center, (12, 2))


@known_bug("B11", "3D rigid transform is not rigid when |n| != 1")
def test_rigid_3d_is_rigid():
    transform = ex.TransformRigid()
    params = np.random.default_rng(0).normal(size=len(transform.init(3)))
    points = np.random.default_rng(1).normal(size=(10, 3))
    new = transform(params, points)
    dist = np.linalg.norm(points[:, None] - points[None], axis=-1)
    newdist = np.linalg.norm(new[:, None] - new[None], axis=-1)
    assert np.allclose(dist, newdist)


@known_bug("B13", "TransformRigid(axis=<ndarray>) raises")
def test_rigid_array_axis():
    assert np.allclose(ex.TransformRigid(axis=np.array([0, 0, 2])).axis, [0, 0, 1])


@known_bug("B14", "the default pyramid goes down to a few pixels")
def test_pyramid_default_levels():
    image = ex.Image(phantom((48, 48, 12)))
    levels, images = ex.Config().setup("Pyramid")(image)
    assert len(levels) <= 4
    assert min(images[0][0].shape) >= 4


@known_bug("B16", "agd keeps rejected steps")
def test_agd_rejected_step():
    def func(x, gradient=True):
        return float(x @ x), 2 * x

    x, _ = ex.agd(func, np.array([1.0]), alpha=10, niter=5)
    assert np.abs(x[0]) <= 1


@known_bug("B18", "Image.__array__ does not accept dtype")
def test_image_asarray_dtype():
    assert np.asarray(ex.Image(np.zeros((4, 4))), dtype=np.float32).dtype == np.float32


@known_bug("B20", "spline0 kernel drops values on bin boundaries")
def test_parzen_spline0_boundary():
    values = np.full(10, 0.5 / 3)  # (nbin - 1) * v = 0.5 for nbin=4
    pdf = ex.parzen2d(values, np.full(10, 0.5), nbin=4)
    assert np.isclose(pdf.sum(), 1)


# tests ported from mutools (test/registration/test_elastix.py)

elastix = ex


@pytest.fixture(autouse=True)
def _seed():
    """the ported tests use the global numpy random state"""
    np.random.seed(0)


def test_image_class():
    arr = np.arange(3 * 4).reshape(3, 4)
    im = elastix.Image(arr)
    assert im.ndim == 2
    assert im.shape == (3, 4)
    assert np.all(im.array == arr)


def test_pyramid_class():
    arr = np.arange(3 * 5).reshape(3, 5)
    im = elastix.Image(arr)

    # dummy
    pyr = elastix.PyramidDummy()
    assert pyr(im) == ([0], [(im,)])

    # nearest neighbour, isotropic
    pyr = elastix.PyramidGaussian(scale_ratio=1.5, order=0, sigma=0, minsize=3)
    levels, reduced = pyr(im)
    assert levels == [1, 0]  # only one reduction was possible
    assert np.allclose(reduced[0][0].shape, (2, 3))
    assert np.allclose(reduced[0][0].array, arr[::2, ::2])
    assert np.allclose(reduced[0][0].spacing, (1.5, 1.5))

    # scaling < 2, spline
    # pyr = elastix.PyramidGaussian(maxlevel=1, scale_ratio=1.2, order=2, sigma=1.5, minsize=1)
    # levels, reduced = pyr(im)
    # assert levels == [1, 0]
    # assert reduced[0][0].shape == (3, 4)  # only one dimension was reduced
    # assert np.all(reduced[0][0].spacing[0] == im.spacing[0])
    # assert np.all(reduced[0][0].spacing[1] > im.spacing[1])

    # nearest neighbour, non isotropic
    im = elastix.Image(arr, spacing=(2, 1))
    pyr = elastix.PyramidGaussian(scale_ratio=2, order=0, sigma=0, minsize=2)
    levels, reduced = pyr(im)
    assert levels == [2, 1, 0]  # only two reductions were possible
    # first level
    assert np.allclose(reduced[1][0].array, arr[::1, ::2])  # axis=0 not reduced
    assert np.allclose(reduced[1][0].spacing, (2, 2))
    # second level
    assert np.allclose(reduced[0][0].array, arr[::2, ::4])
    assert np.allclose(reduced[0][0].spacing, (4, 4))

    # mutiple images with different scaling
    im2 = elastix.Image(arr, spacing=(2, 2), origin=(-1, -1))
    pyr = elastix.PyramidGaussian(
        maxlevel=1, scale_ratio=2, order=0, sigma=0, minsize=1
    )
    levels, reduced = pyr(im, im2)
    assert levels == [1, 0]
    assert reduced[0][0].shape == (3, 3)
    assert reduced[0][1].shape == (3, 5)  # 2nd image was no reduced on axis=1
    assert np.allclose(reduced[0][0].spacing, reduced[0][1].spacing)

    # test mask reduction
    arr = np.arange(5 * 7).reshape(5, 7)
    mask = np.isin(arr, [16, 17, 18])  # keep center
    im3 = elastix.Image(arr, mask=mask)
    pyr = elastix.PyramidGaussian(
        maxlevel=1, scale_ratio=2, order=0, sigma=0, minsize=1
    )
    levels, reduced = pyr(im3)
    assert levels == [1, 0]
    assert reduced[0][0].mask.shape == reduced[0][0].shape
    assert np.all(
        reduced[0][0].mask == np.isin(reduced[0][0].array, [16, 17, 18])
    )  # center


def test_transform_class():

    with pytest.raises(TypeError):
        elastix.Transform()

    #
    # translation transform
    transform = elastix.TransformTranslation()
    assert np.allclose(transform.init(3), np.zeros(3))

    points = np.random.rand(10, 3)
    params = np.array([1, 2, 3])
    moved, jac = transform(params, points, jac=True)
    assert np.allclose(moved, points + params)

    # jacobian
    for i in range(100):
        vec = np.random.uniform(size=3)
        finitediff = (transform(params + 1e-10 * vec, points) - moved) * 1e10
        assert np.allclose(jac @ vec, finitediff, atol=1e-5)

    #
    # Rigid transform

    # null transform 2d
    transform = elastix.TransformRigid()
    points = np.random.rand(10, 2)
    params = transform.init(2)
    assert np.allclose(transform(params, points), points)  # no change

    # 90 degree rotation
    params = np.array([np.pi / 2, 0, 0])
    assert np.allclose(transform(params, [1, 2]), [-2, 1])

    # 180 degree rotation + (-1, -1)
    params = np.array([np.pi, 1, 1])
    assert np.allclose(transform(params, [1, 2]), [0, -1])
    assert np.allclose(transform(params, [2, 1]), [-1, 0])

    # jacobian
    params = np.random.normal(loc=[np.pi, 1, 1])
    moved, jac = transform(params, points, jac=True)
    vec = vec = np.random.uniform(size=3)
    for i in range(100):
        finitediff = (transform(params + 1e-10 * vec, points) - moved) * 1e10
        assert np.allclose(jac @ vec, finitediff, atol=1e-5)

    # centered
    # transform = elastix.TransformRigid(centered=True)
    # moved, jac = transform(params, points, jac=True)
    # for i in range(100):
    #     finitediff = (transform(params + 1e-10 * vec, points) - moved) * 1e10
    #     assert np.allclose(jac @ vec, finitediff, atol=1e-5)

    # inverse
    points = np.random.uniform(size=[10, 2])
    center = np.random.uniform(size=2)
    params = np.random.uniform(low=[-np.pi, -1, -1], high=[np.pi, 1, 1])
    iparams = transform.inverse(params)
    assert np.allclose(
        transform(iparams, transform(params, points, center=center), center=center),
        points,
    )

    # Rigid transform 3d

    # rotation transform 3d
    transform = elastix.TransformRigid()
    points = np.random.rand(10, 3)
    params = transform.init(3)
    assert np.allclose(transform(params, points), points)  # no change

    params = transform.pack(phi=np.pi, n=[1, 0, 0], b=[0, 0, 0])
    assert np.allclose(transform(params, [1, 0, 0]), [1, 0, 0])  # fixed point
    assert np.allclose(transform(params, [0, 1, 1]), [0, -1, -1])  # moving point

    # centered
    transform = elastix.TransformRigid()
    params = transform.pack(phi=np.pi / 2, n=[1, 0, 0], b=[0, 0, 0])
    assert np.allclose(
        transform(params, [[0, 1, 1], [0, -1, 1]], center=[0, 0, 1]),
        [[0, 0, 2], [0, 0, 0]],
    )

    # jacobian
    for i in range(100):
        params = np.random.normal(
            loc=[np.pi] + [i % 3, (i + 1) % 3, (i + 2) % 3] + [0, 0, 0]
        )
        moved, jac = transform(params, points, jac=True)
        vec = np.random.uniform(size=params.size)
        finitediff = (transform(params + 1e-10 * vec, points) - moved) * 1e10
        assert np.allclose(jac @ vec, finitediff, atol=1e-3)

    # centered
    for i in range(100):
        center = np.random.uniform(3)
        params = np.random.normal(
            loc=[np.pi] + [i % 3, (i + 1) % 3, (i + 2) % 3] + [0, 0, 0]
        )
        moved, jac = transform(params, points, jac=True, center=center)
        vec = np.random.uniform(size=params.size)
        finitediff = (
            transform(params + 1e-10 * vec, points, center=center) - moved
        ) * 1e10
        assert np.allclose(jac @ vec, finitediff, atol=1e-3)

    # inverse
    # for i in range(100):
    #     points = np.random.uniform(size=[10,3])
    #     params = np.random.uniform(
    #         low=[-np.pi, -1, -1, -1, -1, -1, -1],
    #         high=[np.pi, 1, 1, 1, 1, 1, 1],
    #     )
    #     iparams = transform.inverse(params)
    #     assert np.allclose(transform(iparams, transform(params, points)), points)

    # fixed axis
    transform = elastix.TransformRigid(axis=[0, 0, 1])
    points = np.random.rand(10, 3)
    params = transform.init(3)
    assert np.allclose(transform(params, points), points)  # no change

    # jacobian
    for i in range(100):
        params = np.random.normal(loc=[np.pi] + [0, 0])  # , 0])
        moved, jac = transform(params, points, jac=True)
        vec = np.random.uniform(size=params.size)
        finitediff = (transform(params + 1e-10 * vec, points) - moved) * 1e10
        assert np.allclose(jac @ vec, finitediff, atol=1e-3)

    #
    # Affine transform
    transform = elastix.TransformAffine()

    # null transform 3d
    params0 = transform.init(3)
    assert np.all(transform.parse(params0)["A"] == np.eye(3))
    assert np.all(transform.parse(params0)["b"] == np.zeros(3))

    points = np.random.rand(10, 3)
    assert np.allclose(transform(params0, points), points)  # no change

    # jacobian
    params = np.random.normal(loc=params0)
    A = transform.parse(params)["A"]
    b = transform.parse(params)["b"]

    moved, jac = transform(params, points, jac=True)
    assert np.allclose(points @ A.T + b, moved)
    for i in range(100):
        vec = np.random.uniform(size=params.size)
        finitediff = (transform(params + 1e-10 * vec, points) - moved) * 1e10
        assert np.allclose(jac @ vec, finitediff, atol=1e-5)

    # Affine transform with fixed axis
    axis = np.random.uniform(size=3)
    axis /= np.linalg.norm(axis)
    transform = elastix.TransformAffine(axis=axis)
    params0 = transform.init(3)
    for i in range(100):
        params = np.random.normal(loc=params0)
        moved, jac = transform(params, points, jac=True)
        vec = np.random.uniform(size=params.size)
        finitediff = (transform(params + 1e-10 * vec, points) - moved) * 1e10
        assert np.allclose(jac @ vec, finitediff, atol=1e-5)
        invp = transform.inverse(params)
        points2 = transform(invp, moved)
        assert np.allclose(points, points2)


def test_transformation_class():
    geometry = elastix.Geometry((10, 12, 12))
    transf = elastix.Transformation("TransformRigid", geometry, [0, 1, 2], axis=2)
    assert np.allclose(transf.transform.axis, [0, 0, 1])
    assert np.allclose(np.asarray(transf), [0, 1, 2])
    # assert np.allclose(transf([[0, 0, 0]]), [[1, 2, 0]])

    # test serialize
    data = transf.serialize()
    assert data["Transform"]["TransformType"] == "TransformRigid"
    assert np.allclose(data["Transform"]["Settings"]["axis"], transf.transform.axis)
    assert data["Transform"]["Settings"]["center"] is None
    assert data["Transform"]["Settings"]["init"] is None
    assert np.allclose(data["Parameters"], [0, 1, 2])
    transf2 = transf.deserialize(data)
    assert np.allclose(transf2.parameters, [0, 1, 2])
    assert transf2.transform.center is None
    assert transf2.transform._init is None

    # test deserialize with wrong parameters
    with pytest.warns(UserWarning):
        transf3 = transf.deserialize(
            {
                **data,
                "Transform": {
                    "Settings": {"center": "None", "init": "None"},
                    "TransformType": "TransformRigid",
                },
            }
        )
    assert np.allclose(transf3.parameters, [0, 1, 2])
    assert transf3.transform.center is None
    assert transf3.transform._init is None


def test_sampler_class():
    with pytest.raises(TypeError):
        elastix.Sampler()

    arr = np.arange(2 * 3).reshape(2, 3)
    im = elastix.Image(arr)

    sampler = elastix.SamplerFull()
    assert np.all(sampler(im)[0] == np.c_[np.nonzero(arr > -1)])


def test_interpolator_class():

    with pytest.raises(TypeError):
        elastix.Interpolator()

    arr = np.arange(3 * 3).reshape(3, 3)
    im = elastix.Image(arr)

    # nearest neighbor
    interpolator = elastix.InterpolatorNearestNeighbor()
    values = interpolator(im, [[0, 0], [0.1, 0.1], [0.9, 0.1], [0, 3]])
    assert np.allclose(values, [0, 0, 3, 2])

    arr = np.arange(10 * 10).reshape(10, 10)
    coords = np.random.uniform(size=(10, 2)) * im.shape

    # linear (+ jacobian)
    interpolator = elastix.InterpolatorLinear(mode="nearest")
    values, jac = interpolator(im, coords, jac=True)
    for i in range(100):
        vec = np.random.uniform(size=coords.shape)
        finitediff = (
            interpolator(im, coords + vec * 1e-10) - interpolator(im, coords)
        ) * 1e10
        assert np.allclose(np.einsum("ijk,jk->i", jac, vec), finitediff, atol=1e-4)

    # order-3 spline
    interpolator = elastix.InterpolatorBSpline(order=3, mode="nearest")
    values, jac = interpolator(im, coords, jac=True)
    for i in range(100):
        vec = np.random.uniform(size=coords.shape)
        finitediff = (
            interpolator(im, coords + vec * 1e-10) - interpolator(im, coords)
        ) * 1e10
        assert np.allclose(np.einsum("ijk,jk->i", jac, vec), finitediff, atol=1e-4)

    # complex
    arr = np.arange(10 * 10).reshape(10, 10)
    arr = arr + 1j * np.arange(10 * 10, 2 * 10 * 10).reshape(10, 10)
    im = elastix.Image(arr)
    interpolator = elastix.InterpolatorBSpline(order=3, mode="nearest")
    coords = np.random.uniform(size=(10, 2), low=4, high=6)
    values, jac = interpolator(im, coords, jac=True)
    for i in range(100):
        vec = np.random.uniform(size=coords.shape)
        finitediff = (
            interpolator(im, coords + vec * 1e-10).real - interpolator(im, coords).real
        ) * 1e10
        assert np.allclose(
            np.einsum("ijk,jk->i", jac, vec),
            np.r_[finitediff, finitediff],
            atol=1e-2,
        )


def test_metric_class():
    base1 = np.repeat(np.random.randint(0, 51, 5), 20)
    base2 = base1 + np.repeat(np.random.randint(-10, 11, 10), 10)
    base3 = np.repeat(np.random.randint(0, 51, 20), 5)
    values1 = base1 + np.random.normal(size=100)
    values2 = base2 + np.random.normal(size=100)
    values3 = base3 + np.random.normal(size=100)

    values1 = elastix.normalize_values(values1)
    values2 = elastix.normalize_values(values2)
    values3 = elastix.normalize_values(values3)
    vec = np.random.uniform(size=values1.shape)

    # MSD
    metric = elastix.MetricMeanSquares()
    assert np.isclose(metric(values1, values1), 0)
    assert (
        metric(values1, values1) < metric(values1, values2) < metric(values1, values3)
    )
    # gradient
    cost, grad = metric(values1, values2, gradient2=True)
    for i in range(100):
        finitediff = (metric(values1, values2 + 1e-10 * vec) - cost) * 1e10
        assert np.isclose(grad @ vec, finitediff, atol=1e-5)

    # Mutual information
    metric = elastix.MetricMutualInformation(nbin=5)
    opts = {"windows": ["spline0", "spline0"]}
    assert (
        metric(values1, values1, **opts)
        < metric(values1, values2, **opts)
        < metric(values1, values3, **opts)
    )

    # gradient
    metric = elastix.MetricMutualInformation(nbin=5)
    cost, grad = metric(values1, values2, gradient2=True)
    for i in range(100):
        finitediff = (metric(values1, values2 + 1e-10 * vec) - cost) * 1e10
        assert np.isclose(grad @ vec, finitediff, atol=1e-5)

    # complex
    ivalues1 = elastix.normalize_values(base1 + np.random.normal(size=100))
    ivalues2 = elastix.normalize_values(base2 + np.random.normal(size=100))
    cvalues1 = values1 + 1j * ivalues1
    cvalues2 = values2 + 1j * ivalues2

    metric = elastix.MetricMeanSquares()
    assert np.isclose(metric(cvalues1, cvalues1), 0)
    assert metric(cvalues1, cvalues1) < metric(cvalues1, cvalues2)

    metric = elastix.MetricMutualInformation(nbin=5)
    assert metric(cvalues1, cvalues1) < metric(cvalues1, cvalues2)

    # gradient
    cost, grad = metric(cvalues1, cvalues2, gradient2=True)
    for i in range(100):
        vec = np.random.uniform(size=cvalues1.shape) + 1j * np.random.uniform(
            size=cvalues1.shape
        )
        finitediff = (metric(cvalues1, cvalues2 + 1e-10 * vec, nbin=5) - cost) * 1e10
        assert np.isclose(grad @ np.r_[vec.real, vec.imag], finitediff, atol=1e-5)


def test_objective_class():
    transform = elastix.TransformRigid()
    sampler = elastix.SamplerFull()
    metric = elastix.MetricMutualInformation()
    interpolator = elastix.InterpolatorLinear()

    arr = np.zeros((20, 20))
    arr[5:15, 5:15] = np.random.randint(10, 20, size=(10, 10))
    fixed = elastix.Image(arr)
    params0 = transform.pack(phi=np.pi / 20, b=[0.2, 0.2])
    moving = fixed.copy(
        values=interpolator(
            fixed, transform(params0, fixed.points, center=fixed.center)
        )
    )

    # normalize values
    moving.update(fixed.normalize(moving.values))
    fixed.update(fixed.normalize())

    objective0 = elastix.Objective(
        fixed, fixed, transform, sampler, metric, interpolator
    )
    objective = elastix.Objective(
        fixed, moving, transform, sampler, metric, interpolator
    )

    assert objective0([0, 0, 0]) < objective([0, 0, 0])
    assert np.isclose(
        objective0(params0, normalize=False), objective([0, 0, 0], normalize=False)
    )

    for i in range(100):
        params = 1e-2 * np.random.uniform(low=[-np.pi, -1, -1], high=[np.pi, 1, 1])
        assert objective0([0, 0, 0]) < objective0(params)

    sol = transform.inverse(params0)
    for i in range(100):
        params = sol + np.random.uniform(low=[-np.pi, -1, -1], high=[np.pi, 1, 1])
        assert objective(sol) < objective(params)

    # gradient
    points1, values1 = sampler(fixed)
    points2, jtrans = transform(params0, points1, jac=True, center=fixed.center)
    values2, jinterp = interpolator(fixed, points2, jac=True)
    ninterp = np.linalg.norm(values2)
    jinterp_trans = np.einsum("ijk,jkl->il", jinterp, jtrans)

    nvalues1 = fixed.normalize(values1)
    nvalues2 = fixed.normalize(values2)
    cost, gmetric = metric(nvalues1, nvalues2, gradient2=True)
    gmetric_interp_trans = gmetric @ jinterp_trans
    cost, grad = objective0(params0, gradient=True)
    for i in range(100):
        vec = np.random.uniform(low=[-np.pi, -1, -1], high=[np.pi, 1, 1])

        # test objective's components separately first
        # transform
        _points2 = transform(params0 + 1e-10 * vec, points1, center=fixed.center)
        finitediff = (_points2 - points2) * 1e10
        assert np.allclose(jtrans @ vec, finitediff, atol=1e-3)

        # interpolator
        _values2 = interpolator(fixed, _points2)
        finitediff = (_values2 - values2) * 1e10
        assert np.allclose(
            jinterp_trans @ vec / ninterp, finitediff / ninterp, atol=1e-4
        )

        # metric
        _nvalues2 = fixed.normalize(_values2)
        _cost = metric(nvalues1, _nvalues2)
        finitediff = (_cost - cost) * 1e10
        assert np.allclose(gmetric_interp_trans @ vec, finitediff, atol=1e-3)

        # objective
        finitediff = (objective0(params0 + 1e-10 * vec) - cost) * 1e10
        assert np.isclose(grad @ vec, finitediff, atol=1e-3)


#
# def test_optimizer_class():
#     arr = np.arange(3 * 3).reshape(3, 3)
#     im1 = elastix.Image(arr)
#
#     arr2 = np.c_[arr - 1, arr[:, -1]]
#     im2 = elastix.Image(arr2)
#
#     sampler = elastix.SamplerFull()
#     interpolator = elastix.InterpolatorBSpline()
#     transform = elastix.TransformTranslation()
#     costfunction = elastix.MetricMeanSquares()
#
#     def objective(parameters, fixed=im1, moving=im2):
#         points1, values1 = sampler(fixed)
#         points2 = transform(parameters, points1)
#         values2 = interpolator(moving, points2)
#         return costfunction(values1, values2)
#
#     optim = elastix.OptimizerGeneric()
#     init = transform.init()
#     params, res = optim(objective, init)
#     # the correct translation was found
#     assert np.allclose(params, [0, 1])


def test_config_class():
    config = elastix.Config()

    assert isinstance(config.setup("Transform"), elastix.Transform)
    assert isinstance(config.setup("Sampler"), elastix.Sampler)
    assert isinstance(config.setup("Optimizer"), elastix.Optimizer)
    assert isinstance(config.setup("Interpolator"), elastix.Interpolator)
    assert isinstance(config.setup("Metric"), elastix.Metric)
    assert isinstance(config.setup("Pyramid"), elastix.Pyramid)

    # update default
    assert isinstance(config.setup("Transform"), elastix.TransformRigid)
    config = elastix.Config({"Transform": "TransformAffine"})
    assert isinstance(config.setup("Transform"), elastix.TransformAffine)

    # update settings
    config["Transform"] = "TransformRigid"
    assert isinstance(config.setup("Transform"), elastix.TransformRigid)


def test_pipeline_class():

    config = elastix.Config(
        {
            "Optimizer": "OptimizerGeneric",
            "Pyramid": "PyramidDummy",
            "Sampler": "SamplerFull",
            "Metric": "MetricMeanSquares",
            "Transform": "TransformRigid",
            "Interpolator": "InterpolatorLinear",
            # options
            "InterpolatorLinear": {"mode": "nearest"},
        }
    )
    pipeline = elastix.Elastix(config)

    arr = np.arange(3 * 3).reshape(3, 3)
    im1 = elastix.Image(arr)

    arr2 = np.c_[arr - 1, arr[:, -1]]
    im2 = elastix.Image(arr2)

    tr = elastix.Transformation("TransformRigid", im1.geometry, [0, 0, 1])
    assert np.allclose(arr, pipeline.apply(tr, im2), atol=1e-5)

    # register images
    params, info = pipeline.execute(im1, im2)
    assert np.allclose(
        params, [0, 0, 1], atol=1e-5
    )  # no rotation, translation = (0, 1)


def test_mutualinfo():

    base1 = np.random.randint(1, 10, size=10)
    values1 = np.ravel(base1 + 2 * np.random.uniform(size=100)[:, np.newaxis])
    values2 = np.ravel(base1 + 2 * np.random.uniform(size=100)[:, np.newaxis])

    values1 = elastix.normalize_values(values1)
    values2 = elastix.normalize_values(values2)

    # parzen windows 1d and 2d
    dist1 = elastix.parzen1d(values1, nbin=10, window="spline0")
    dist2 = elastix.parzen1d(values2, nbin=10, window="spline3")
    dist12 = elastix.parzen2d(values1, values2, nbin=10, windows=["spline0", "spline3"])
    assert np.allclose(dist12.sum(axis=1), dist1, atol=1e-5)
    assert np.allclose(dist12.sum(axis=0), dist2, atol=1e-5)

    # gradient
    opts = dict(nbin=10, window="spline3")
    dist2, jac = elastix.parzen1d(values2, jac=True, **opts)
    vec = np.random.uniform(size=len(values2))
    finitediff = (elastix.parzen1d(values2 + 1e-10 * vec, **opts) - dist2) * 1e10
    assert np.allclose(jac @ vec, finitediff, atol=1e-5)

    opts = dict(nbin=10, windows=["spline0", "spline3"])
    dist12, jac2 = elastix.parzen2d(values1, values2, jac2=True, **opts)
    finitediff = (
        elastix.parzen2d(values1, values2 + 1e-10 * vec, **opts) - dist12
    ) * 1e10
    assert np.allclose(jac2 @ vec, finitediff, atol=1e-5)
    assert np.allclose(jac2.sum(0), jac)

    # mutual info
    opts = dict(windows=["spline3", "spline3"], nbin=10)

    mi11, grad11 = elastix.mutualinfo(values1, values1, gradient2=True, **opts)
    mi12, grad12 = elastix.mutualinfo(values1, values2, gradient2=True, **opts)
    assert mi11 < mi12
    finitediff = (
        elastix.mutualinfo(values1, values1 + 1e-10 * vec, **opts) - mi11
    ) * 1e10
    assert np.allclose(grad11 @ vec, finitediff, atol=1e-5)

    finitediff = (
        elastix.mutualinfo(values1, values2 + 1e-10 * vec, **opts) - mi12
    ) * 1e10
    assert np.allclose(grad12 @ vec, finitediff, atol=1e-5)


def _test_adg():
    def classifier(obs, theta):
        return 1 / (1 + np.exp(-theta @ obs))

    def dclassifier(obs, theta):
        return obs * np.exp(-theta @ obs) * classifier(obs, theta) ** 2

    def loss(pred, gt, eps=1e-10):
        costs = -gt * np.log(pred + eps) - (1 - gt) * np.log(1 - pred + eps)
        cost = costs.sum()
        grad = -gt / (pred + eps) + (1 - gt) / (1 - pred + eps)
        return cost, grad

    # train data
    theta0 = np.array([1, -2])
    rstate = np.random.RandomState(0)
    x_ = rstate.uniform(-2, 2, size=(2, 200))
    y_ = classifier(x_, theta0)
    gt = np.rint(y_)
    obs = x_ + 1e-1 * rstate.normal(size=(2, 200))

    assert np.isclose(loss(gt, gt)[0], 0, atol=1e-5)

    def costfun(theta):
        pred = classifier(obs, theta)
        gclass = dclassifier(obs, theta)
        cost, gloss = loss(pred, gt)
        grad = gclass @ gloss
        return cost, grad

    # check costfunction calculation
    # cost, grad = costfun(theta)
    # for i in range(100):
    #     delta = np.random.normal(size=2)
    #     fdiff = (costfun(theta + 1e-10 * delta)[0] - cost) * 1e10
    #     assert np.isclose(fdiff, grad @ delta, atol=1e-4)

    minloss = loss(y_, gt)[0]

    # solve
    init = np.zeros(2)
    sol, info = elastix.agd(costfun, init)

    assert info["fun"] < minloss

    pred = classifier(obs, sol)
    crate = np.mean(np.rint(pred) == gt)
    # print(f"classification rate: {crate}")
    assert crate > 0.9

    # from matplotlib import pyplot as plt
    # print(f"Pred: {sol}, GT: {theta0}")
    # plt.figure()
    # plt.plot(info["values"])
    # plt.hlines(minloss, *plt.xlim(), color="k")
    # plt.show()

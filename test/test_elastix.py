""" tests for tomio.elastix

Regression tests of the bugs fixed in phase 1 are labelled with their id in PLAN_elastix.md (B1, B2...).
"""
import numpy as np
import pytest

import tomio as io
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


def grid_points(shape, spacing=None, origin=None, direction=None):
    """world coordinates of the pixels of a grid"""
    geometry = ex.Geometry(shape, spacing, origin, direction)
    return geometry.points


def world_phantom(points, box=40.0, seed=0, nblob=12):
    """sum of gaussian blobs defined in world coordinates (in [0, box]^ndim)"""
    rng = np.random.default_rng(seed)
    ndim = points.shape[1]
    centers = rng.uniform(0.2, 0.8, size=(nblob, ndim)) * box
    widths = rng.uniform(0.08, 0.15, size=(nblob, 1)) * box
    weights = rng.uniform(0.5, 1, size=nblob)
    dist2 = (((points[:, np.newaxis] - centers) / widths) ** 2).sum(axis=-1)
    return (weights * np.exp(-0.5 * dist2)).sum(axis=1)


def sample_phantom(shape, spacing=None, origin=None, direction=None, shift=0):
    """world phantom sampled on a grid, translated by `shift` (moving(y) = fixed(y - shift))"""
    points = grid_points(shape, spacing, origin, direction)
    return world_phantom(points - np.asarray(shift)).reshape(shape)


def inner_error(registered, fixed, margin=6):
    """mean absolute difference away from the borders"""
    inner = tuple(slice(margin, -margin) if n > 2 * margin else slice(None) for n in fixed.shape)
    return np.abs(np.asarray(registered) - fixed)[inner].mean()


def make_objective(fixed, moving, config=None, spacing=None):
    config = ex.Config(config)
    return ex.Objective(
        ex.Image(fixed, spacing=spacing),
        ex.Image(moving, spacing=spacing),
        config.setup("transform"),
        config.setup("sampler"),
        config.setup("metric"),
        config.setup("interpolator"),
    )


def finite_gradient(func, params, step=1e-5):
    params = np.asarray(params, dtype=float)
    return np.array([(func(params + step * e) - func(params - step * e)) / (2 * step) for e in np.eye(len(params))])


TRANSLATION = {"transform": "translation"}
# stochastic optimizer of the B-spline tests: fewer iterations than the default (fast tests)
SGD = {"type": "sgd", "niter": 64}
MI = {"metric": "mi"}


# registration

@pytest.mark.parametrize("shape, shift", [((48, 48), (2.5, -1.5)), ((32, 32, 16), (1.5, -1.0, 0.5))])
def test_translation_mse(shape, shift):
    fixed, moving = shifted(shape, shift)
    registered, trans = ex.register(fixed, moving, config=TRANSLATION, return_transform=True)
    assert isinstance(trans, ex.Transformation)
    assert np.allclose(trans.parameters, shift, atol=0.05)
    assert inner_error(registered, fixed) < 1e-2


# B10 (fixed): the rotation center changes between pyramid levels: translation not recovered
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


@pytest.mark.parametrize("transform", ["translation", "rigid", "affine"])
def test_serialize_roundtrip(transform):
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    _, trans = ex.register(fixed, moving, transform=transform, return_transform=True)
    trans2 = ex.Transformation.deserialize(trans.serialize())
    assert type(trans2.transform) is type(trans.transform)
    assert np.allclose(trans2.parameters, trans.parameters)
    assert np.allclose(trans2(), trans())


@pytest.mark.parametrize(
    "config, params",
    [
        (TRANSLATION, [0.3, 0.2]),
        ({"transform": "rigid"}, [0.01, 0.3, 0.2]),
        ({"transform": "affine"}, [1.01, 0.02, -0.01, 0.99, 0.3, 0.2]),
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
    for section in ex.COMPONENTS:
        assert config.setup(section) is not None
        assert config[section]["type"] == ex.Config.DEFAULTS[section]
    with pytest.raises(ValueError):
        config.setup("unknown")


def test_config_sections():
    # type name, dict with type and options, options only
    config = ex.Config(transform="affine", metric={"type": "mi", "nbin": 16})
    assert isinstance(config.setup("transform"), ex.TransformAffine)
    assert config.setup("metric").nbin == 16
    config["metric"] = {"nbin": 8}  # keep type
    assert config["metric"] == {"type": "mi", "nbin": 8}
    config["metric"] = "mse"  # new type: default options
    assert config["metric"] == {"type": "mse"}
    # dict and keywords
    config = ex.Config({"transform": "translation", "metric": "mi"}, metric="mse")
    assert config["transform"]["type"] == "translation" and config["metric"]["type"] == "mse"
    # copy
    assert ex.Config(config) == config


def test_config_component_object():
    config = ex.Config(transform=ex.TransformRigid(axis=2))
    transform = config.setup("transform")
    assert isinstance(transform, ex.TransformRigid) and np.allclose(transform.axis, [0, 0, 1])
    # a new object at each setup (components may hold state)
    assert config.setup("transform") is not transform


@pytest.mark.parametrize(
    "sections, message",
    [
        ({"foo": "rigid"}, "Unknown section"),
        ({"transform": "foo"}, "Unknown transform type"),
        ({"metric": {"type": "mi", "nbins": 32}}, "Unknown option"),
        ({"metric": {"nbin": 32}}, "Unknown option"),  # default metric (mse) has no option
        ({"resample": {"order": 3}}, "Unknown resample option"),
    ],
)
def test_config_validation(sections, message):
    with pytest.raises(ValueError, match=message):
        ex.Config(sections)


def test_config_file(tmp_path):
    from tomio import configfile

    configfile.write(tmp_path / "config.yml", {"transform": {"type": "rigid", "axis": 2}, "metric": "mi"})
    config = ex.Config(tmp_path / "config.yml")
    assert config["transform"] == {"type": "rigid", "axis": 2}
    assert config["metric"]["type"] == "mi"
    # stages
    configfile.write(tmp_path / "stages.yml", [{"transform": "translation"}, {"transform": "rigid"}])
    stages = ex.make_stages(tmp_path / "stages.yml")
    assert [stage["transform"]["type"] for stage in stages] == ["translation", "rigid"]


def test_config_describe():
    text = ex.Config.describe()
    for section, types in ex.COMPONENTS.items():
        assert f"{section}:" in text
        for name in types:
            assert f"    {name}:" in text
    assert "nbin = 32" in text


def test_config_str():
    import yaml

    config = ex.Config(transform={"type": "rigid", "axis": 2}, metric="mi")
    assert ex.Config(yaml.safe_load(str(config))) == config


def test_register_keywords():
    fixed, moving = shifted((40, 40), (1.5, 1.0))
    _, trans = ex.register(fixed, moving, transform="translation", metric="mi", return_transform=True)
    assert isinstance(trans.transform, ex.TransformTranslation)
    assert np.allclose(trans.parameters, (1.5, 1.0), atol=0.05)
    with pytest.raises(ValueError):
        ex.register(fixed, moving, [TRANSLATION, TRANSLATION], metric="mi")


def test_register_stages():
    fixed, moving = rotated((48, 48), 0.05, (2.5, -1.5))
    registered, trans = ex.register(fixed, moving, [TRANSLATION, {"transform": "rigid"}], return_transform=True)
    assert isinstance(trans, ex.TransformationSequence) and len(trans) == 2
    assert isinstance(trans[0].transform, ex.TransformTranslation)
    assert isinstance(trans[1].transform, ex.TransformRigid)
    assert inner_error(registered, fixed) < 0.1 * inner_error(moving, fixed)
    # composition: second stage applied after the first
    points = np.random.default_rng(0).uniform(0, 40, size=(5, 2))
    assert np.allclose(trans(points), trans[1](trans[0](points)))
    # serialization
    data = trans.serialize()
    assert isinstance(data, list) and len(data) == 2
    trans2 = ex.as_transformation(data)
    assert np.allclose(trans2(points), trans(points))
    assert np.allclose(np.asarray(ex.transform(data, moving)), np.asarray(registered))


# regression tests (PLAN_elastix.md, phase 1)

# B1 (fixed): Elastix.execute ignores `init`
def test_execute_uses_init():
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    config = ex.Config({**TRANSLATION, "pyramid": "none", "optimizer": {"type": "lbfgs", "maxiter": 0}})
    trans, _ = ex.Elastix(config).execute(ex.Image(fixed), ex.Image(moving), init=[0.7, 0.3])
    assert np.allclose(trans.parameters, [0.7, 0.3])


# B2 (fixed): transform(ref=...) uses `ref.geom`
def test_transform_with_ref():
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    _, trans = ex.register(fixed, moving, config=TRANSLATION, return_transform=True)
    out = ex.transform(trans, moving, ref=ex.Image(np.zeros((20, 24))))
    assert np.asarray(out).shape == (20, 24)


# B3 (fixed): serialize uses the bound method `self.init`
@pytest.mark.parametrize("transform, init", [("rigid", [0.1, 1.0, 2.0]), ("affine", [1, 0, 0, 1, 1.0, 2.0])])
def test_serialize_with_init(transform, init):
    trans = ex.Transformation({"type": transform, "init": np.asarray(init)}, ex.Geometry((8, 8)), init)
    data = ex.Transformation.deserialize(trans.serialize()).transform.serialize()
    assert np.allclose(data["init"], init)


# B4 (fixed): TransformTranslation.serialize drops `axis`
def test_serialize_translation_axis():
    transform = ex.Transform.deserialize(ex.TransformTranslation(axis=2).serialize())
    assert transform.axis == 2


# B5 (fixed): linear interpolation gradient lacks the 1/spacing factor
def test_gradient_anisotropic_spacing():
    spacing = (1.0, 1.0, 5.0)
    fixed, moving = shifted((24, 24, 12), (1.0, -1.0, 2.0), spacing=spacing)
    objective = make_objective(fixed, moving, TRANSLATION, spacing=spacing)
    params = [0.3, 0.2, 0.7]
    _, gradient = objective(params, gradient=True)
    assert np.allclose(gradient, finite_gradient(objective, params), rtol=1e-2, atol=1e-6)


# B6 (fixed): MI gradient lacks the intensity normalization factor
def test_gradient_mi_intensity_scale():
    fixed, moving = shifted((40, 40), (1.5, 1.0), scale=100)
    objective = make_objective(fixed, moving, {**TRANSLATION, **MI})
    params = [0.3, 0.2]
    _, gradient = objective(params, gradient=True)
    assert np.allclose(gradient, finite_gradient(objective, params), rtol=1e-2, atol=1e-6)


# B7 (fixed): B-spline interpolator gradient has the wrong shape
def test_register_bspline_interpolator():
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    _, trans = ex.register(fixed, moving, config={**TRANSLATION, "interpolator": "bspline"}, return_transform=True)
    assert np.allclose(trans.parameters, (1.0, 0.5), atol=0.05)


# B8 (fixed): out-of-FOV moving samples count as 0
def test_cost_out_of_fov():
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    objective = make_objective(fixed, moving, TRANSLATION)
    assert objective([1000.0, 0.0]) == np.inf  # no valid sample


# B8 (fixed): out-of-FOV samples make the cost discontinuous at the image border (full sampling)
def test_register_full_sampler():
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    _, trans = ex.register(fixed, moving, config={**TRANSLATION, "sampler": "full"}, return_transform=True)
    assert np.allclose(trans.parameters, (1.0, 0.5), atol=0.05)


# B10 (fixed): Geometry.center is off by half a voxel
def test_geometry_center():
    assert np.allclose(ex.Geometry((3, 5), spacing=(2, 1), origin=(10, 0)).center, (12, 2))


# B11 (fixed): 3D rigid transform is not rigid when |n| != 1
def test_rigid_3d_is_rigid():
    transform = ex.TransformRigid()
    params = np.random.default_rng(0).normal(size=len(transform.init(3)))
    points = np.random.default_rng(1).normal(size=(10, 3))
    new = transform(params, points)
    dist = np.linalg.norm(points[:, None] - points[None], axis=-1)
    newdist = np.linalg.norm(new[:, None] - new[None], axis=-1)
    assert np.allclose(dist, newdist)


# B13 (fixed): TransformRigid(axis=<ndarray>) raises
def test_rigid_array_axis():
    assert np.allclose(ex.TransformRigid(axis=np.array([0, 0, 2])).axis, [0, 0, 1])


# B14 (fixed): the default pyramid goes down to a few pixels
def test_pyramid_default_levels():
    image = ex.Image(phantom((48, 48, 12)))
    levels, images = ex.Config().setup("pyramid")(image)
    assert len(levels) <= 4
    assert min(images[0][0].shape) >= 4


# B18 (fixed): Image.__array__ does not accept dtype
def test_image_asarray_dtype():
    assert np.asarray(ex.Image(np.zeros((4, 4))), dtype=np.float32).dtype == np.float32


# B20 (fixed): spline0 kernel drops values on bin boundaries
def test_parzen_spline0_boundary():
    values = np.full(10, 0.5 / 3)  # (nbin - 1) * v = 0.5 for nbin=4
    pdf = ex.parzen2d(values, np.full(10, 0.5), nbin=4)
    assert np.isclose(pdf.sum(), 1)


GRADIENT_TRANSFORMS = {
    "translation": [0.3, 0.2, 0.7],
    "rigid": [0.02, -0.01, 0.03, 0.3, 0.2, 0.7],
    "affine": [1.01, 0.02, -0.01, 0.01, 0.99, 0.02, -0.02, 0.01, 1.02, 0.3, 0.2, 0.7],
}


# B5, B6, B7 (fixed): gradients of all components, anisotropic spacing and intensity range
@pytest.mark.parametrize("transform", list(GRADIENT_TRANSFORMS))
@pytest.mark.parametrize("interpolator", ["linear", "bspline"])
@pytest.mark.parametrize("metric", ["mse", "mi"])
def test_gradient_components(transform, interpolator, metric):
    spacing = (1.0, 1.5, 4.0)
    fixed, moving = shifted((16, 16, 8), (1.0, -1.0, 2.0), spacing=spacing, scale=50)
    config = {"transform": transform, "interpolator": interpolator, "metric": metric}
    objective = make_objective(fixed, moving, config, spacing=spacing)
    params = GRADIENT_TRANSFORMS[transform]
    _, gradient = objective(params, gradient=True)
    assert np.allclose(gradient, finite_gradient(objective, params), rtol=1e-2, atol=1e-6)


# B23 (fixed): oblique grids
def test_register_oblique():
    angle = 0.4
    direction = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    origin, shift = (20, -5), (1.5, -1.0)
    fixed = sample_phantom((40, 40), origin=origin, direction=direction)
    moving = sample_phantom((40, 40), origin=origin, direction=direction, shift=shift)
    im_fixed = ex.Image(fixed, origin=origin, direction=direction)
    im_moving = ex.Image(moving, origin=origin, direction=direction)
    registered, trans = ex.register(im_fixed, im_moving, config=TRANSLATION, return_transform=True)
    # translation in world coordinates
    assert np.allclose(trans.parameters, shift, atol=0.05)
    assert np.allclose(registered.direction, direction)
    assert inner_error(registered, fixed) < 1e-2


def test_image_from_volume():
    """tomio Volume: transform = rows of axis directions (columns of the ITK direction matrix)"""
    direction = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    vol = io.Volume(np.zeros((4, 5, 6)), spacing=(1, 2, 3), origin=(1, 1, 1), transform=direction.T)
    image = ex.Image(vol)
    assert np.allclose(image.direction, direction)
    assert np.allclose(image.geometry.to_points([[1, 1, 1]]), vol.get_coords([1, 1, 1]))
    # tomio convention
    assert np.allclose(image.transform, vol.transform)


def test_image_keeps_mask():
    image = ex.Image(np.zeros((4, 4)), mask=np.eye(4))
    assert np.array_equal(ex.Image(image).mask, np.eye(4, dtype=bool))


# B10 (fixed): the transform found at a coarse level is applied on the full resolution grid
def test_register_minlevel():
    fixed, moving = rotated((48, 48), 0.05, (2.5, -1.5))
    config = {**MI, "pyramid": {"minlevel": 1}}
    registered = ex.register(fixed, moving, config=config)
    assert inner_error(registered, fixed) < 0.2 * inner_error(moving, fixed)


# B2, B10 (fixed): transform(ref=...) keeps the center of rotation
def test_transform_ref_keeps_center():
    fixed, moving = rotated((40, 40), 0.1, (1.5, -1.0))
    registered, trans = ex.register(fixed, moving, return_transform=True)
    ref = ex.Image(np.zeros((20, 30)), origin=(5, 10))
    out = ex.transform(trans, moving, ref=ref)
    assert np.allclose(np.asarray(out), np.asarray(registered)[5:25, 10:40], atol=1e-6)


# B8 (fixed): fixed image larger than the moving image (moving: thick slab)
def test_register_moving_slab():
    shift = (1.5, -1.0, 2.0)
    fixed = sample_phantom((32, 32, 32))
    slab = dict(spacing=(1, 1, 4), origin=(0, 0, 10))
    moving = sample_phantom((32, 32, 4), shift=shift, **slab)
    _, trans = ex.register(ex.Image(fixed), ex.Image(moving, **slab), config=TRANSLATION, return_transform=True)
    # (accuracy limited by the linear interpolation between the 4 mm slices of the moving image)
    assert np.allclose(trans.parameters, shift, atol=0.15)


# B8 (fixed): fixed image is a thick slab inside the moving image
def test_register_fixed_slab():
    shift = (1.5, -1.0, 2.0)
    slab = dict(spacing=(1, 1, 6), origin=(0, 0, 8))
    fixed = sample_phantom((32, 32, 4), **slab)
    moving = sample_phantom((32, 32, 32), shift=shift)
    # samples on the fixed grid (random coordinates interpolate the fixed image between thick slices)
    config = {**TRANSLATION, **MI, "sampler": "random"}
    _, trans = ex.register(ex.Image(fixed, **slab), ex.Image(moving), config=config, return_transform=True)
    assert np.allclose(trans.parameters, shift, atol=0.05)


# transform(): methods, global transforms through tomio.resample

def rigid_transformation(geometry, params=(0.02, -0.01, 0.05, 1.5, -1.0, 2.0)):
    return ex.Transformation("rigid", geometry, np.asarray(params, dtype=float))


def test_transformation_affine():
    trans = rigid_transformation(ex.Geometry((10, 12, 8), spacing=(1, 1, 3)))
    A, t = trans.affine()
    points = np.random.default_rng(0).uniform(0, 10, size=(5, 3))
    assert np.allclose(points @ A.T + t, trans(points))
    assert np.allclose(A, ex.euler_to_matrix(trans.parameters[:3]))


@pytest.mark.parametrize("method, order", [("linear", 1), ("spline3", 3)])
def test_transform_global_3d(method, order):
    """global 3d transforms: tomio.resample on the mapped grid == interpolation at the transformed points"""
    spacing = (1.0, 1.0, 3.0)
    moving = ex.Image(sample_phantom((24, 24, 10), spacing=spacing), spacing=spacing)
    ref = ex.Image(np.zeros((20, 22, 8)), spacing=(1.2, 1.0, 3.5), origin=(1, 2, 3))
    trans = rigid_transformation(ref.geometry)
    out = ex.transform(trans, moving, ref=ref, method=method)
    expected = ex.InterpolatorBSpline(order=order, mode="constant")(moving, trans()).reshape(out.shape)
    inner = (slice(4, -4), slice(4, -4), slice(2, -2))
    assert np.allclose(np.asarray(out)[inner], expected[inner], atol=1e-4)
    assert np.allclose(out.spacing, ref.spacing) and np.allclose(out.origin, ref.origin)


def test_transform_roi_sparse_slices():
    """ROI labelled on every 4th slice: 'roi' method of tomio.resample"""
    roi = np.zeros((20, 20, 24), dtype=np.uint8)
    roi[5:15, 5:15, ::4] = 1
    roi[8:12, 8:12, ::4] = 2
    roi = io.Volume(roi, spacing=(1, 1, 5))
    # target slices between the labelled slices (z = 0, 20, 40...)
    target = io.Volume(np.zeros((20, 20, 4)), spacing=(1, 1, 20), origin=(0, 0, 10))
    identity = ex.Transformation("rigid", ex.Image(target).geometry, np.zeros(6))
    out = ex.transform(identity, roi, ref=target, method="roi")
    assert np.array_equal(np.asarray(out), io.resample(roi, like=target, method="roi"))
    assert set(np.unique(out)) == {0, 1, 2}
    # nearest neighbour misses the labelled slices
    assert set(np.unique(ex.transform(identity, roi, ref=target, method="nearest"))) == {0}


def test_transform_default_method():
    labels = np.zeros((16, 16, 8), dtype=np.uint8)
    labels[4:12, 4:12, 2:6] = 3
    trans = rigid_transformation(ex.Geometry(labels.shape))
    out = ex.transform(trans, labels)
    assert np.asarray(out).dtype == np.uint8
    assert set(np.unique(out)) <= {0, 3}


def test_transform_roi_2d():
    """non-global path (2d): warp_roi"""
    roi = np.zeros((8, 8), dtype=np.uint8)
    roi[2:6, 3:5] = 2
    trans = ex.Transformation("translation", ex.Geometry((8, 8)), [1.0, 0.0])
    out = ex.transform(trans, roi, method="roi")
    assert np.array_equal(np.asarray(out)[1:5, 3:5], np.full((4, 2), 2))
    assert np.asarray(out).sum() == 2 * 8

def test_transform_complex_3d():
    """complex images: own interpolation (not tomio.resample)"""
    image = sample_phantom((16, 16, 8)) * (1 + 1j)
    trans = rigid_transformation(ex.Geometry(image.shape), params=np.zeros(6))
    out = ex.transform(trans, image, method="linear")
    assert np.iscomplexobj(np.asarray(out))
    assert np.allclose(np.asarray(out), image)


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
    pyr = elastix.PyramidGaussian(scale_ratio=1.5, order=1, sigma=0, minsize=2, maxlevel=1)
    levels, reduced = pyr(im)
    assert levels == [1, 0]
    assert np.allclose(reduced[0][0].shape, (2, 3))
    # centered grid with spacing 1.5 (arr is linear: exact interpolation)
    rows, cols = np.array([0.25, 1.75]), np.array([0.5, 2, 3.5])
    assert np.allclose(reduced[0][0].array, 5 * rows[:, None] + cols)
    assert np.allclose(reduced[0][0].spacing, (1.5, 1.5))
    # same center
    assert np.allclose(reduced[0][0].center, im.center)

    # minsize: minimum size of the reduced axes (axis 0 is not reduced)
    pyr = elastix.PyramidGaussian(scale_ratio=1.5, order=1, sigma=0, minsize=3)
    levels, reduced = pyr(im)
    assert levels == [1, 0]
    assert reduced[0][0].shape == (3, 3)
    assert np.allclose(reduced[0][0].array, 5 * np.arange(3)[:, None] + cols)

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
        finitediff = (transform(params + 1e-6 * vec, points) - transform(params - 1e-6 * vec, points)) / 2e-6
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
        finitediff = (transform(params + 1e-6 * vec, points) - transform(params - 1e-6 * vec, points)) / 2e-6
        assert np.allclose(jac @ vec, finitediff, atol=1e-5)

    # centered
    # transform = elastix.TransformRigid(centered=True)
    # moved, jac = transform(params, points, jac=True)
    # for i in range(100):
    #     finitediff = (transform(params + 1e-6 * vec, points) - transform(params - 1e-6 * vec, points)) / 2e-6
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

    # Euler angles: rotation of pi about x
    params = transform.pack(angles=[np.pi, 0, 0], b=[0, 0, 0])
    assert np.allclose(transform(params, [1, 0, 0]), [1, 0, 0])  # fixed point
    assert np.allclose(transform(params, [0, 1, 1]), [0, -1, -1])  # moving point

    # same rotation matrix as ITK's Euler3DTransform: R = Rz @ Rx @ Ry
    angles = [0.3, -0.5, 0.7]
    R = elastix.euler_to_matrix(angles)
    assert np.allclose(transform(np.r_[angles, 0, 0, 0], np.eye(3)), R.T)
    assert np.allclose(elastix.matrix_to_euler(R), angles)

    # centered
    transform = elastix.TransformRigid()
    params = transform.pack(angles=[np.pi / 2, 0, 0], b=[0, 0, 0])
    assert np.allclose(
        transform(params, [[0, 1, 1], [0, -1, 1]], center=[0, 0, 1]),
        [[0, 0, 2], [0, 0, 0]],
    )

    # jacobian
    for i in range(100):
        params = np.random.normal(loc=[np.pi, 0, 0, 0, 0, 0])
        moved, jac = transform(params, points, jac=True)
        vec = np.random.uniform(size=params.size)
        finitediff = (transform(params + 1e-6 * vec, points) - transform(params - 1e-6 * vec, points)) / 2e-6
        assert np.allclose(jac @ vec, finitediff, atol=1e-3)

    # centered
    for i in range(100):
        center = np.random.uniform(size=3)
        params = np.random.normal(loc=[np.pi, 0, 0, 0, 0, 0])
        moved, jac = transform(params, points, jac=True, center=center)
        vec = np.random.uniform(size=params.size)
        finitediff = (
            transform(params + 1e-6 * vec, points, center=center)
            - transform(params - 1e-6 * vec, points, center=center)
        ) / 2e-6
        assert np.allclose(jac @ vec, finitediff, atol=1e-3)

    # inverse
    for i in range(100):
        points = np.random.uniform(size=[10, 3])
        center = np.random.uniform(size=3)
        params = np.random.uniform(low=[-np.pi, -np.pi / 2, -np.pi, -1, -1, -1], high=[np.pi, np.pi / 2, np.pi, 1, 1, 1])
        iparams = transform.inverse(params)
        moved = transform(params, points, center=center)
        assert np.allclose(transform(iparams, moved, center=center), points)

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
        finitediff = (transform(params + 1e-6 * vec, points) - transform(params - 1e-6 * vec, points)) / 2e-6
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
        finitediff = (transform(params + 1e-6 * vec, points) - transform(params - 1e-6 * vec, points)) / 2e-6
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
        finitediff = (transform(params + 1e-6 * vec, points) - transform(params - 1e-6 * vec, points)) / 2e-6
        assert np.allclose(jac @ vec, finitediff, atol=1e-5)
        invp = transform.inverse(params)
        points2 = transform(invp, moved)
        assert np.allclose(points, points2)


def test_transformation_class():
    geometry = elastix.Geometry((10, 12, 12))
    transf = elastix.Transformation({"type": "rigid", "axis": 2}, geometry, [0, 1, 2])
    assert np.allclose(transf.transform.axis, [0, 0, 1])
    assert np.allclose(np.asarray(transf), [0, 1, 2])
    # assert np.allclose(transf([[0, 0, 0]]), [[1, 2, 0]])

    # test serialize
    data = transf.serialize()
    assert data["transform"]["type"] == "rigid"
    assert np.allclose(data["transform"]["axis"], transf.transform.axis)
    assert data["transform"]["center"] is None
    assert data["transform"]["init"] is None
    assert np.allclose(data["parameters"], [0, 1, 2])
    transf2 = transf.deserialize(data)
    assert np.allclose(transf2.parameters, [0, 1, 2])
    assert transf2.transform.center is None
    assert transf2.transform._init is None



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
            interpolator(im, coords + vec * 1e-7) - interpolator(im, coords)
        ) * 1e7
        assert np.allclose(np.einsum("ijk,jk->i", jac, vec), finitediff, atol=1e-4)

    # order-3 spline
    interpolator = elastix.InterpolatorBSpline(order=3, mode="nearest")
    values, jac = interpolator(im, coords, jac=True)
    for i in range(100):
        vec = np.random.uniform(size=coords.shape)
        finitediff = (
            interpolator(im, coords + vec * 1e-7) - interpolator(im, coords)
        ) * 1e7
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
            interpolator(im, coords + vec * 1e-7).real - interpolator(im, coords).real
        ) * 1e7
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

    # (inside (0, 1): central finite differences)
    interior = lambda values: 0.005 + 0.99 * elastix.normalize_values(values)
    values1 = interior(values1)
    values2 = interior(values2)
    values3 = interior(values3)
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
        finitediff = (metric(values1, values2 + 1e-6 * vec) - metric(values1, values2 - 1e-6 * vec)) / 2e-6
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
        finitediff = (metric(values1, values2 + 1e-6 * vec) - metric(values1, values2 - 1e-6 * vec)) / 2e-6
        assert np.isclose(grad @ vec, finitediff, atol=1e-5)

    # complex
    ivalues1 = interior(base1 + np.random.normal(size=100))
    ivalues2 = interior(base2 + np.random.normal(size=100))
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
        finitediff = (metric(cvalues1, cvalues2 + 1e-6 * vec, nbin=5) - metric(cvalues1, cvalues2 - 1e-6 * vec, nbin=5)) / 2e-6
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

    # sample in the center: the transformed samples stay inside the moving image
    fixed.mask = arr > 0

    # (moving image without mask: the mask of `fixed` restricts the samples only)
    objective0 = elastix.Objective(
        fixed, fixed.copy(mask=None), transform, sampler, metric, interpolator
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
        step = 1e-6
        points2p = transform(params0 + step * vec, points1, center=fixed.center)
        points2m = transform(params0 - step * vec, points1, center=fixed.center)
        finitediff = (points2p - points2m) / (2 * step)
        assert np.allclose(jtrans @ vec, finitediff, atol=1e-3)

        # interpolator
        values2p, values2m = interpolator(fixed, points2p), interpolator(fixed, points2m)
        finitediff = (values2p - values2m) / (2 * step)
        assert np.allclose(
            jinterp_trans @ vec / ninterp, finitediff / ninterp, atol=1e-4
        )

        # metric
        costp = metric(nvalues1, fixed.normalize(values2p))
        costm = metric(nvalues1, fixed.normalize(values2m))
        finitediff = (costp - costm) / (2 * step)
        assert np.allclose(gmetric_interp_trans @ vec, finitediff, atol=1e-3)

        # objective
        finitediff = (objective0(params0 + step * vec) - objective0(params0 - step * vec)) / (2 * step)
        assert np.isclose(grad @ vec, finitediff, atol=1e-3)


def test_config_class():
    config = elastix.Config()

    assert isinstance(config.setup("transform"), elastix.Transform)
    assert isinstance(config.setup("sampler"), elastix.Sampler)
    assert isinstance(config.setup("optimizer"), elastix.Optimizer)
    assert isinstance(config.setup("interpolator"), elastix.Interpolator)
    assert isinstance(config.setup("metric"), elastix.Metric)
    assert isinstance(config.setup("pyramid"), elastix.Pyramid)

    # update default
    assert isinstance(config.setup("transform"), elastix.TransformRigid)
    config = elastix.Config({"transform": "affine"})
    assert isinstance(config.setup("transform"), elastix.TransformAffine)

    # update settings
    config["transform"] = "rigid"
    assert isinstance(config.setup("transform"), elastix.TransformRigid)


def test_pipeline_class():

    config = elastix.Config(
        {
            "optimizer": "lbfgs",
            "pyramid": "none",
            "sampler": "full",
            "metric": "mse",
            "transform": "rigid",
            "interpolator": {"type": "linear", "mode": "nearest"},
        }
    )
    pipeline = elastix.Elastix(config)

    arr = np.arange(3 * 3).reshape(3, 3)
    im1 = elastix.Image(arr)

    arr2 = np.c_[arr - 1, arr[:, -1]]
    im2 = elastix.Image(arr2)

    tr = elastix.Transformation("rigid", im1.geometry, [0, 0, 1])
    assert np.allclose(arr, np.asarray(elastix.transform(tr, im2, method="linear")), atol=1e-5)

    # register images
    params, info = pipeline.execute(im1, im2)
    # no rotation, translation = (0, 1) (optimizer tolerance: 1e-3 mm)
    assert np.allclose(params, [0, 0, 1], atol=1e-3)


def test_mutualinfo():

    base1 = np.random.randint(1, 10, size=10)
    values1 = np.ravel(base1 + 2 * np.random.uniform(size=100)[:, np.newaxis])
    values2 = np.ravel(base1 + 2 * np.random.uniform(size=100)[:, np.newaxis])

    # (inside (0, 1): central finite differences)
    values1 = 0.005 + 0.99 * elastix.normalize_values(values1)
    values2 = 0.005 + 0.99 * elastix.normalize_values(values2)

    # parzen windows 2d: joint distribution
    dist12 = elastix.parzen2d(values1, values2, nbin=10, windows=["spline0", "spline3"])
    assert np.isclose(dist12.sum(), 1)
    # spline0 window: marginal = histogram
    hist = np.bincount(np.floor(values1 * 9 + 0.5).astype(int), minlength=10) / len(values1)
    assert np.allclose(dist12.sum(axis=1), hist)

    # gradient
    vec = np.random.uniform(size=len(values2))

    opts = dict(nbin=10, windows=["spline0", "spline3"])
    dist12, jac2 = elastix.parzen2d(values1, values2, jac2=True, **opts)
    finitediff = (
        elastix.parzen2d(values1, values2 + 1e-6 * vec, **opts) - elastix.parzen2d(values1, values2 - 1e-6 * vec, **opts)
    ) / 2e-6
    assert np.allclose(jac2 @ vec, finitediff, atol=1e-5)

    # mutual info
    opts = dict(windows=["spline3", "spline3"], nbin=10)

    mi11, grad11 = elastix.mutualinfo(values1, values1, gradient2=True, **opts)
    mi12, grad12 = elastix.mutualinfo(values1, values2, gradient2=True, **opts)
    assert mi11 < mi12
    finitediff = (
        elastix.mutualinfo(values1, values1 + 1e-6 * vec, **opts) - elastix.mutualinfo(values1, values1 - 1e-6 * vec, **opts)
    ) / 2e-6
    assert np.allclose(grad11 @ vec, finitediff, atol=1e-5)

    finitediff = (
        elastix.mutualinfo(values1, values2 + 1e-6 * vec, **opts) - elastix.mutualinfo(values1, values2 - 1e-6 * vec, **opts)
    ) / 2e-6
    assert np.allclose(grad12 @ vec, finitediff, atol=1e-5)


# masks

def test_register_masks():
    fixed, moving = shifted((48, 48), (1.5, 1.0))
    # corrupted images, 4 pixels away from the masks
    # (interpolation and smoothing mix the values on each side of the mask border)
    mask = np.zeros((48, 48), dtype=bool)
    mask[12:36, 12:36] = True
    clean = np.zeros((48, 48), dtype=bool)
    clean[8:40, 8:40] = True
    noise = np.random.default_rng(0).uniform(size=(48, 48))
    moving_bad = np.where(clean, moving, noise)
    fixed_bad = np.where(clean, fixed, noise)
    options = dict(transform="translation", return_transform=True)
    _, trans = ex.register(fixed_bad, moving_bad, fixed_mask=mask, moving_mask=mask, **options)
    assert np.allclose(trans.parameters, (1.5, 1.0), atol=0.05)
    # without masks
    _, trans = ex.register(fixed_bad, moving_bad, **options)
    assert not np.allclose(trans.parameters, (1.5, 1.0), atol=0.05)


def test_objective_moving_mask():
    fixed, moving = shifted((20, 20), (0.0, 0.0))
    mask = np.zeros((20, 20), dtype=bool)
    mask[:, :10] = True
    config = ex.Config(transform="translation", sampler="full")
    objective = ex.Objective(
        ex.Image(fixed), ex.Image(moving, mask=mask, spacing=(2, 2)), config.setup("transform"),
        config.setup("sampler"), config.setup("metric"), config.setup("interpolator"),
    )
    # moving mask in moving coordinates (spacing 2): pixels 0..9 along axis 1, i.e. y < 19 mm
    points = np.c_[np.full(5, 10.0), [4.2, 17.2, 18.6, 19.4, 30.2]]
    valid = objective.valid(points, np.zeros(len(points)))
    assert valid.tolist() == [True, True, True, False, False]


def test_results_types():
    """3d results: tomio Volume with the fixed geometry; 2d results: Image"""
    direction = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    fixed = io.Volume(sample_phantom((20, 20, 10)), spacing=(1, 1, 2), origin=(1, 2, 3), transform=direction.T)
    registered, trans = ex.register(fixed, fixed, transform="translation", return_transform=True)
    assert isinstance(registered, io.Volume)
    assert registered.geometry.same_as(fixed.geometry)
    assert isinstance(ex.transform(trans, fixed), io.Volume)
    registered2d = ex.register(np.ones((8, 8)), np.ones((8, 8)), transform="translation")
    assert isinstance(registered2d, ex.Image)


# phase 3: no scipy dependency

def test_without_scipy():
    import subprocess
    import sys

    code = """
import sys
sys.modules["scipy"] = None
import numpy as np
from tomio import elastix
points = np.indices((24, 24, 12)).reshape(3, -1).T
centers = np.random.default_rng(0).uniform(4, [20, 20, 8], size=(6, 3))
def blobs(points):
    return np.exp(-0.5 * (((points[:, None] - centers) / 3) ** 2).sum(-1)).sum(1).reshape(24, 24, 12)
fixed, moving = blobs(points), blobs(points - [2, 0, 0])
reg, trans = elastix.register(fixed, moving, transform="translation", metric="mi", return_transform=True)
assert np.allclose(trans.parameters, [2, 0, 0], atol=0.1), trans.parameters
elastix.transform(trans, (fixed > 0.5).astype(np.uint8), method="roi")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("shape", [(13, 17), (9, 7, 5), (8, 1, 6)])
def test_spline_matches_scipy(order, shape):
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(order)
    arr = rng.random(shape)
    coords = rng.uniform(0, np.array(shape) - 1, size=(200, len(shape)))
    if order == 0:
        coords = np.round(coords) + rng.uniform(-0.4, 0.4, size=coords.shape)  # away from the rounding ties
        coords = np.clip(coords, 0, np.array(shape) - 1)
    values = ex.ndspline_interpolation(arr, coords, order=order, mode="mirror")
    expected = ndimage.map_coordinates(arr, coords.T, order=order, mode="mirror")
    assert np.allclose(values, expected, atol=1e-8)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_spline_gradient(order):
    rng = np.random.default_rng(order)
    arr = rng.random((9, 8, 6))
    coords = rng.uniform(-1, np.array(arr.shape), size=(100, 3))  # some outside (mode nearest: zero derivative)
    _, jac = ex.ndspline_interpolation(arr, coords, order=order, mode="nearest", jac=True)
    grad = np.diagonal(jac, axis1=0, axis2=1).T
    step = 1e-6
    for d in range(3):
        delta = step * np.eye(3)[d]
        fd = ex.ndspline_interpolation(arr, coords + delta, order=order, mode="nearest")
        fd -= ex.ndspline_interpolation(arr, coords - delta, order=order, mode="nearest")
        fd /= 2 * step
        # (linear: kinks on the grid nodes)
        smooth = np.abs(coords[:, d] - np.round(coords[:, d])) > 1e-4 if order == 1 else slice(None)
        assert np.allclose(grad[smooth, d], fd[smooth], atol=1e-5)


def test_spline_modes():
    arr = np.arange(12.0).reshape(3, 4)  # arr[i, j] = 4 i + j
    coords = np.array([[-1.0, 0.0], [1.0, 4.5], [1.0, 1.0]])
    assert np.allclose(ex.ndspline_interpolation(arr, coords, order=1, mode="constant", cval=-1), [-1, -1, 5])
    assert np.allclose(ex.ndspline_interpolation(arr, coords, order=1, mode="nearest"), [0, 7, 5])
    # mirror: row -1 -> 1, column 4.5 -> 1.5
    assert np.allclose(ex.ndspline_interpolation(arr, coords, order=1, mode="mirror"), [4, 5.5, 5])
    _, jac = ex.ndspline_interpolation(arr, coords, order=1, mode="mirror", jac=True)
    assert np.allclose(np.diagonal(jac, axis1=0, axis2=1).T, [[-4, 1], [4, -1], [4, 1]])


def test_gaussian_filter_and_dilation():
    ndimage = pytest.importorskip("scipy.ndimage")
    arr = np.zeros((31, 25))
    arr[15, 12] = 1
    smoothed = ex.gaussian_filter(arr, (2.0, 0.0))
    assert np.isclose(smoothed.sum(), 1, atol=1e-3)
    # (SimpleITK's discrete gaussian kernel differs slightly from scipy's sampled gaussian)
    assert np.allclose(smoothed[:, 12], ndimage.gaussian_filter1d(arr[:, 12], 2.0), atol=1e-2)
    assert np.allclose(smoothed[:, [11, 13]], 0)  # no smoothing along axis 1
    mask = np.zeros((7, 8), dtype=bool)
    mask[3, 4] = True
    assert np.array_equal(ex.binary_dilation(mask), ndimage.binary_dilation(mask))


def test_lbfgs_rosenbrock():
    def rosenbrock(x):
        cost = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        grad = np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])
        return cost, grad

    x, info = ex.lbfgs(rosenbrock, [-1.2, 1.0], maxiter=200, gtol=1e-10, ftol=0, xtol=0)
    assert info["success"], info["message"]
    assert np.allclose(x, [1, 1], atol=1e-5)


# phase 4: accelerations

@pytest.mark.parametrize("nbin", [8, 32])
def test_mutualinfo_scatter(nbin):
    """scatter-add MI == dense Parzen windows MI (values and gradients)"""
    rng = np.random.default_rng(nbin)
    values1 = rng.uniform(size=500)
    values2 = np.clip(values1 + 0.1 * rng.normal(size=500), 0, 1)
    values1[:5] = [0, 1, 0.5 / (nbin - 1), 1.5 / (nbin - 1), 0.25]  # edges and bin boundaries
    values2[:5] = [0, 1, 0.5, 1 / (nbin - 1), 0.75]
    opts = dict(nbin=nbin, windows=["spline0", "spline3"])
    p12, jac = ex.parzen2d(values1, values2, jac2="split", **opts)
    log12 = ex._log_ratio(p12)
    expected = -(p12 * log12).sum()
    expected_grad = -np.einsum("ik,jk,ij->k", *jac, log12)
    mi, grad = ex.mutualinfo_scatter(values1, values2, nbin=nbin, gradient2=True)
    assert np.isclose(mi, expected)
    assert np.allclose(grad, expected_grad)
    assert np.isclose(ex.mutualinfo(values1, values2, **opts), expected)


def test_sampler_thick_axes():
    image = ex.Image(np.zeros((20, 20, 5)), spacing=(1, 1, 6))
    points, _ = ex.SamplerRandomCoordinates(maxsample=200)(image)
    coords = image.geometry.to_coords(points)
    assert np.allclose(coords[:, 2], np.round(coords[:, 2]))  # thick axis: on the grid
    assert not np.allclose(coords[:, 0], np.round(coords[:, 0]))
    points, _ = ex.SamplerRandomCoordinates(maxsample=200, thick_ratio=None)(image)
    assert not np.allclose(image.geometry.to_coords(points)[:, 2] % 1, 0)


def test_register_fixed_slab_default_sampler():
    """thick-slice fixed image with the default sampler (random coordinates in-plane only)"""
    shift = (1.5, -1.0, 2.0)
    slab = dict(spacing=(1, 1, 6), origin=(0, 0, 8))
    fixed = sample_phantom((32, 32, 4), **slab)
    moving = sample_phantom((32, 32, 32), shift=shift)
    _, trans = ex.register(ex.Image(fixed, **slab), ex.Image(moving), transform="translation", metric="mi", return_transform=True)
    assert np.allclose(trans.parameters, shift, atol=0.05)


@pytest.mark.parametrize("method", ["geometry", "moments"])
def test_register_initialize(method):
    """large offset between the images: recovered with the initialization"""
    shift = np.array([12.0, -9.0])
    fixed = sample_phantom((40, 40), shift=(0, 0))
    moving = sample_phantom((40, 40), origin=shift, shift=shift)  # same content, shifted grid
    options = dict(transform="translation", return_transform=True)
    _, trans = ex.register(ex.Image(fixed), ex.Image(moving, origin=shift), initialize=method, **options)
    assert np.allclose(trans.parameters, shift, atol=0.05)
    trans_init = ex.initialize(ex.TransformTranslation(), ex.Image(fixed), ex.Image(moving, origin=shift), method)
    assert np.allclose(trans_init, shift, atol=0.5 if method == "moments" else 1e-8)


def test_execute_levels_info():
    fixed, moving = shifted((48, 48), (1.5, 1.0))
    trans, info = ex.Elastix(ex.Config(transform="translation")).execute(ex.Image(fixed), ex.Image(moving))
    assert len(info.levels) >= 2
    assert [res.level for res in info.levels] == sorted([res.level for res in info.levels], reverse=True)
    for res in info.levels:
        assert res.nfev >= 1 and np.isfinite(res.fun) and res.nsample > 0
    assert np.allclose(info.levels[-1].parameters, trans.parameters)


def test_parameter_scales():
    radius = 50.0
    assert np.allclose(ex.TransformRigid().scales(np.zeros(6), radius), [50, 50, 50, 1, 1, 1])
    assert np.allclose(ex.TransformRigid().scales(np.zeros(3), radius), [50, 1, 1])
    assert np.allclose(ex.TransformAffine().scales(np.zeros(6), radius), [50, 50, 50, 50, 1, 1])
    assert np.allclose(ex.TransformTranslation().scales(np.zeros(3), radius), [1, 1, 1])
    geometry = ex.Geometry((10, 20), spacing=(2, 1))
    assert np.isclose(geometry.radius(), np.sqrt((20**2 + 20**2) / 12))


# phase 5: free-form deformation (B-spline)

def smooth_displacement(points, amplitude=1.5, period=24.0):
    """known smooth displacement field u(x)"""
    u = np.zeros_like(points)
    u[:, 0] = amplitude * np.sin(2 * np.pi * points[:, 1] / period)
    u[:, 1] = amplitude * np.cos(2 * np.pi * points[:, 0] / period)
    return u


def deformed_pair(shape=(48, 48), **kwargs):
    """fixed(x) = moving(x + u(x)): T(x) = x + u(x)"""
    points = grid_points(shape)
    moving = world_phantom(points).reshape(shape)
    fixed = world_phantom(points + smooth_displacement(points, **kwargs)).reshape(shape)
    return fixed, moving


def bspline_objective(fixed, moving, metric="mse", bending=0.0, grid_spacing=8.0):
    config = ex.Config(transform={"type": "bspline", "grid_spacing": grid_spacing, "bending": bending}, metric=metric)
    objective = make_objective(fixed, moving, config)
    objective.transform.setup(objective.fixed.geometry)
    return objective


def test_bspline_grid():
    transform = ex.TransformBSpline(grid_spacing=10)
    geometry = ex.Geometry((41, 21, 1), spacing=(1, 2, 5), origin=(5, 0, 0))
    transform.setup(geometry)
    grid = transform.grid
    assert grid["shape"].tolist() == [7, 7, 4]  # 4 + 3 control points (40 mm / 10), single-slice axis: 1 + 3
    assert np.allclose(grid["spacing"][:2], [10, 10])
    assert np.allclose(grid["origin"][:2], [-5, -10])
    params = np.zeros(transform.init(3).size)
    assert params.size == 3 * 7 * 7 * 4
    points = geometry.points
    assert np.allclose(transform(params, points), points)
    # constant displacement: B-splines sum to 1
    params = np.repeat([1.0, -2.0, 0.5], transform.ncontrol)
    assert np.allclose(transform(params, points), points + [1.0, -2.0, 0.5])


@pytest.mark.parametrize("metric", ["mse", "mi", "ngf"])
@pytest.mark.parametrize("bending", [0.0, 50.0])
def test_bspline_gradient(metric, bending):
    fixed, moving = deformed_pair((32, 32))
    objective = bspline_objective(fixed, moving * 50, metric=metric, bending=bending)
    rng = np.random.default_rng(0)
    params = rng.normal(scale=0.5, size=objective.transform.init(2).size)
    _, gradient = objective(params, gradient=True)
    check = rng.choice(params.size, 12, replace=False)
    step = 1e-5
    for k in check:
        delta = np.zeros(params.size)
        delta[k] = step
        fd = (objective(params + delta) - objective(params - delta)) / (2 * step)
        assert np.isclose(gradient[k], fd, rtol=1e-2, atol=1e-7), (k, gradient[k], fd)


def test_bspline_penalty():
    transform = ex.TransformBSpline(grid_spacing=5, bending=2.0)
    transform.setup(ex.Geometry((20, 16, 12)))
    rng = np.random.default_rng(1)
    params = rng.normal(size=transform.init(3).size)
    energy, grad = transform.penalty(params, gradient=True)
    assert energy > 0
    # quadratic: E(p) = p.grad / 2
    assert np.isclose(energy, params @ grad / 2)
    # affine displacements have no bending energy
    grid = transform.grid
    coords = np.stack(np.meshgrid(*[np.arange(n) for n in grid["shape"]], indexing="ij"), -1).reshape(-1, 3)
    affine = (coords @ rng.normal(size=(3, 3))).T.ravel()
    assert np.isclose(transform.penalty(affine), 0, atol=1e-12)


def textured_pair(shape=(48, 48), amplitude=1.0, period=32.0, noise=0.02, seed=0):
    """textured noisy images (as MRI), aligned up to a small smooth deformation: fixed(x) = moving(x + u(x))

    (smooth noise-free phantoms are degenerate for MI and NGF with free-form deformations: intensity plateaus
    let the deformation "improve" the histograms or edge magnitudes)
    """
    rng = np.random.default_rng(seed)
    points = grid_points(shape)
    extent = np.array(shape, dtype=float)
    nblob = int(np.prod(shape) / 14)
    centers = rng.uniform(2, extent - 2, size=(nblob, 2))
    widths = rng.uniform(1.0, 3.0, size=(nblob, 1))
    weights = rng.uniform(0.2, 1, size=nblob)

    def texture(p):
        dist2 = (((p[:, np.newaxis] - centers) / widths) ** 2).sum(axis=-1)
        return (weights * np.exp(-0.5 * dist2)).sum(axis=1).reshape(shape)

    u = smooth_displacement(points, amplitude=amplitude, period=period)
    moving = texture(points) + noise * rng.normal(size=shape)
    fixed = texture(points + u) + noise * rng.normal(size=shape)
    return fixed, moving, points, u


@pytest.mark.parametrize(
    "metric, contrast, tolerance",
    [("mse", False, 0.2), ("mi", False, 0.2), ("mi", True, 0.25), ("ngf", False, 0.25), ("ngf", True, 0.25),
     ("mind", False, 0.25), ("mind", True, 0.35)],
)
def test_register_bspline(metric, contrast, tolerance):
    """recovery of a known small smooth deformation (images aligned beforehand, e.g. by a rigid stage)"""
    fixed, moving, points, u = textured_pair()
    if contrast:
        moving = np.exp(-1.5 * moving)  # other contrast (non-linear, decreasing)
    registered, trans = ex.register(fixed, moving, transform={"type": "bspline", "grid_spacing": 8}, metric=metric, optimizer=SGD, return_transform=True)
    inner = np.all((points >= 8) & (points <= 39), axis=1)
    error = np.linalg.norm(trans(points[inner]) - points[inner] - u[inner], axis=1)
    assert error.mean() < tolerance * np.linalg.norm(u[inner], axis=1).mean()


@pytest.mark.parametrize("metric", ["mse", "mi", "ngf", "mind"])
def test_register_bspline_identical_textured(metric):
    """identical images: (almost) no deformation"""
    fixed, _, points, _ = textured_pair()
    _, trans = ex.register(fixed, fixed, transform={"type": "bspline", "grid_spacing": 8}, metric=metric, optimizer=SGD, return_transform=True)
    assert np.linalg.norm(trans(points) - points, axis=1).mean() < 0.1


def test_register_bspline_identical():
    """identical images: no deformation (MSE)"""
    image = sample_phantom((40, 40))
    config = {"transform": {"type": "bspline", "grid_spacing": 8}, "optimizer": "lbfgs"}  # deterministic
    _, trans = ex.register(image, image, config, return_transform=True)
    assert np.abs(trans.parameters).max() < 1e-3


def test_register_rigid_bspline_stages():
    fixed, moving = deformed_pair((48, 48), amplitude=1.0)
    moving = np.roll(moving, 2, axis=0)  # + global shift
    stages = [{"transform": "rigid"}, {"transform": {"type": "bspline", "grid_spacing": 8}, "optimizer": SGD}]
    registered, trans = ex.register(fixed, moving, stages, return_transform=True)
    assert isinstance(trans, ex.TransformationSequence)
    assert isinstance(trans[1].transform, ex.TransformBSpline)
    assert not trans.is_global
    assert inner_error(registered, fixed, margin=8) < 0.2 * inner_error(moving, fixed, margin=8)


def test_bspline_serialize():
    fixed, moving = deformed_pair((32, 32))
    _, trans = ex.register(fixed, moving, transform={"type": "bspline", "grid_spacing": 8}, optimizer=SGD, return_transform=True)
    data = trans.serialize()
    assert data["transform"]["type"] == "bspline" and data["transform"]["grid"] is not None
    trans2 = ex.as_transformation(data)
    points = grid_points((32, 32))
    assert np.allclose(trans2(points), trans(points))
    assert np.allclose(np.asarray(ex.transform(data, moving)), np.asarray(ex.transform(trans, moving)))


def test_warp_roi_matches_tomio():
    """warp_roi (backward, any transform) vs tomio.resample(method="roi") (identity)"""
    roi = np.zeros((20, 20, 24), dtype=np.uint8)
    roi[5:15, 5:15, ::4] = 1
    roi[8:12, 8:12, ::4] = 2
    roi = io.Volume(roi, spacing=(1, 1, 5))
    for spacing, origin in [((1, 1, 20), (0, 0, 10)), ((1, 1, 30), (0, 0, 7.5)), ((2, 2, 5), (0.5, 0.5, 0))]:
        target = io.Volume(np.zeros((10, 10, 4)), spacing=spacing, origin=origin)
        geometry = ex.Image(target).geometry
        out = ex.warp_roi(ex.Image(roi), geometry.points, geometry).reshape(target.shape)
        expected = io.resample(roi, like=target, method="roi")
        agreement = np.mean(out == np.asarray(expected))
        assert agreement > 0.97, (spacing, agreement)


def test_warp_labels():
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[4:12, 4:8] = 1
    labels[4:12, 8:12] = 2
    image = ex.Image(labels)
    out = ex.warp_labels(image, image.geometry.points).reshape(labels.shape)
    assert np.array_equal(out, labels)
    out = ex.warp_labels(image, image.geometry.points + [0, 0.4]).reshape(labels.shape)
    assert set(np.unique(out)) == {0, 1, 2}


@pytest.mark.parametrize("transform, params", [("translation", [0.3, -0.2]), ("rigid", [0.02, 0.3, -0.2]), ("affine", [1.01, 0.02, -0.01, 0.99, 0.3, -0.2])])
def test_gradient_ngf(transform, params):
    fixed, moving, _, _ = textured_pair((32, 32))
    # (B-spline interpolation of the gradient images: smooth cost for the finite differences)
    config = {"transform": transform, "metric": "ngf", "interpolator": "bspline"}
    objective = make_objective(fixed, np.exp(-1.5 * moving), config)
    _, gradient = objective(params, gradient=True)
    assert np.allclose(gradient, finite_gradient(objective, params), rtol=1e-3, atol=1e-8)


def test_ngf_metric():
    rng = np.random.default_rng(0)
    g1 = rng.normal(size=(50, 2))
    metric = ex.MetricNormalizedGradientFields()
    assert metric(g1, g1, eps=(1e-6, 1e-6)) < 1e-6  # aligned
    assert np.isclose(metric(g1, -3 * g1, eps=(1e-6, 1e-6)), 0, atol=1e-6)  # polarity and scale invariant
    g2 = np.c_[-g1[:, 1], g1[:, 0]]  # orthogonal
    assert np.isclose(metric(g1, g2, eps=(1e-6, 1e-6)), 1)
    cost, grad = metric(g1, g2 + 0.1, gradient2=True, eps=(0.1, 0.2))
    delta = 1e-6 * rng.normal(size=g2.shape)
    assert np.isclose(metric(g1, g2 + 0.1 + delta, eps=(0.1, 0.2)) - cost, np.sum(grad * delta), rtol=1e-3)


# stochastic gradient descent

def test_register_sgd_translation():
    fixed, moving = shifted((48, 48), (2.5, -1.5))
    _, trans = ex.register(fixed, moving, transform="translation", metric="mi", optimizer="sgd", return_transform=True)
    assert np.allclose(trans.parameters, (2.5, -1.5), atol=0.1)


def test_sgd_resamples():
    """new samples at each iteration"""
    fixed, moving = shifted((32, 32), (1.0, 0.5))
    config = ex.Config(transform="translation", sampler={"maxsample": 100})
    objective = make_objective(fixed, moving, config)
    first = objective.sampler(objective.fixed)[0].copy()
    objective.resample()
    assert not np.allclose(objective.sampler(objective.fixed)[0], first)


@pytest.mark.parametrize("metric", ["mse", "mi"])
def test_register_bspline_sgd(metric):
    fixed, moving, points, u = textured_pair()
    config = {"transform": {"type": "bspline", "grid_spacing": 8}, "metric": metric, "optimizer": {"type": "sgd", "niter": 64}, "pyramid": {"maxlevel": 1}}
    _, trans = ex.register(fixed, moving, config, return_transform=True)
    inner = np.all((points >= 8) & (points <= 39), axis=1)
    error = np.linalg.norm(trans(points[inner]) - points[inner] - u[inner], axis=1)
    assert error.mean() < 0.3 * np.linalg.norm(u[inner], axis=1).mean()
    # identical images: small drift
    _, trans = ex.register(fixed, fixed, config, return_transform=True)
    assert np.linalg.norm(trans(points[inner]) - points[inner], axis=1).mean() < 0.1


# plotting (optional matplotlib)

@pytest.mark.parametrize("deformable", [False, True])
def test_plot_registration(tmp_path, deformable):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from tomio import elastix_plot

    fixed, moving = shifted((24, 24, 6), (1.0, -1.0, 0.0))
    if deformable:
        stages = [{"transform": "translation"}, {"transform": {"type": "bspline", "grid_spacing": 8}, "optimizer": {"type": "sgd", "niter": 8}}]
    else:
        stages = {"transform": "translation"}
    _, trans = ex.register(fixed, moving, stages, return_transform=True)
    fig = elastix_plot.plot_registration(fixed, moving, trans, slices=2)
    assert len(fig.axes) >= 2 * 6
    fig.savefig(tmp_path / "registration.png")
    assert (tmp_path / "registration.png").exists()
    _, info = ex.Elastix(ex.Config(transform="translation")).execute(ex.Image(fixed), ex.Image(moving))
    elastix_plot.plot_levels(info).savefig(tmp_path / "levels.png")


def test_local_displacement():
    from tomio import elastix_plot

    geometry = ex.Geometry((8, 8))
    rigid = ex.Transformation("translation", geometry, [1.0, 2.0])
    points = geometry.points
    assert np.allclose(elastix_plot.local_displacement(rigid, points), 0)
    bspline = ex.TransformBSpline(grid_spacing=4)
    bspline.setup(geometry)
    params = np.repeat([0.5, -0.5], bspline.ncontrol)
    sequence = ex.TransformationSequence([rigid, ex.Transformation(bspline, geometry, params)])
    assert np.allclose(elastix_plot.local_displacement(sequence, points), [0.5, -0.5])


# B-spline with a fixed axis (in-plane deformation)

def test_bspline_fixed_axis():
    transform = ex.TransformBSpline(grid_spacing=10, axis=2)
    geometry = ex.Geometry((20, 20, 6), spacing=(1, 1, 4))
    transform.setup(geometry)
    params = np.random.default_rng(0).normal(size=transform.init(3).size)
    assert params.size == 2 * transform.ncontrol
    points = geometry.points
    moved = transform(params, points)
    assert np.allclose(moved[:, 2], points[:, 2])  # slices stay slices
    assert not np.allclose(moved[:, :2], points[:, :2])
    # serialization
    trans = ex.Transformation(transform, geometry, params)
    trans2 = ex.as_transformation(trans.serialize())
    assert trans2.transform.axis == 2 and np.allclose(trans2(points), trans(points))


@pytest.mark.parametrize("metric", ["mse", "mi"])
def test_bspline_fixed_axis_gradient(metric):
    spacing = (1.0, 1.0, 3.0)
    fixed, moving = shifted((16, 16, 8), (1.0, -1.0, 0.0), spacing=spacing, scale=50)
    config = {"transform": {"type": "bspline", "grid_spacing": 6, "axis": 2}, "metric": metric}
    objective = make_objective(fixed, moving, config, spacing=spacing)
    objective.transform.setup(objective.fixed.geometry)
    rng = np.random.default_rng(1)
    params = rng.normal(scale=0.3, size=objective.transform.init(3).size)
    _, gradient = objective(params, gradient=True)
    for k in rng.choice(params.size, 8, replace=False):
        delta = np.zeros(params.size)
        delta[k] = 1e-5
        fd = (objective(params + delta) - objective(params - delta)) / 2e-5
        assert np.isclose(gradient[k], fd, rtol=1e-2, atol=1e-7)


def test_register_bspline_fixed_axis():
    """in-plane deformation only: no displacement along z"""
    fixed, moving, u, points = None, None, None, None
    shape, spacing = (32, 32, 6), (1.0, 1.0, 4.0)
    points = grid_points(shape, spacing)
    u = np.zeros_like(points)
    u[:, :2] = smooth_displacement(points[:, :2], amplitude=1.0, period=32.0)
    fixed = world_phantom(points + u, box=32).reshape(shape)
    moving = world_phantom(points, box=32).reshape(shape)
    config = {"transform": {"type": "bspline", "grid_spacing": 8, "axis": 2}, "optimizer": SGD}
    _, trans = ex.register(ex.Image(fixed, spacing=spacing), ex.Image(moving, spacing=spacing), config, return_transform=True)
    moved = trans(points)
    assert np.allclose(moved[:, 2], points[:, 2])


def test_warp_roi_whole_slices():
    """a labelled slice midway between two output slices goes to one of them only; tilted mapping: no split"""
    roi = np.zeros((16, 16, 12), dtype=np.uint8)
    roi[4:12, 4:12, 6] = 1  # z = 30 mm
    roi = ex.Image(roi, spacing=(1, 1, 5))
    target = ex.Geometry((16, 16, 4), spacing=(1, 1, 10), origin=(0, 0, 5))  # slices at z = 5, 15, 25, 35, ...
    out = ex.warp_roi(roi, target.points, target).reshape(target.shape)
    labelled = [k for k in range(4) if out[..., k].any()]
    assert len(labelled) == 1 and out[..., labelled[0]].sum() == 64
    # small tilt of the mapping: still one output slice, whole
    trans = ex.Transformation("rigid", target, [0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    out = ex.warp_roi(roi, trans(), target).reshape(target.shape)
    assert len([k for k in range(4) if out[..., k].any()]) == 1


def test_bending_grid_independent():
    """the bending penalty of a given smooth displacement hardly depends on the grid spacing"""
    geometry = ex.Geometry((100, 100, 40), spacing=(1, 1, 2.5))

    def field(points):
        u = np.zeros_like(points)
        u[:, 0] = 3 * np.sin(2 * np.pi * points[:, 1] / 80)
        u[:, 1] = 3 * np.cos(2 * np.pi * points[:, 0] / 80) * np.sin(2 * np.pi * points[:, 2] / 120)
        return u

    penalties = []
    for spacing in [20, 10, 5]:
        transform = ex.TransformBSpline(grid_spacing=spacing, bending=1.0)
        transform.setup(geometry)
        shape = tuple(transform.grid["shape"])
        disp = field(transform.control_points())
        coeffs = np.concatenate([ex.bspline_coefficients(disp[:, a].reshape(shape), 3).ravel() for a in range(3)])
        penalties.append(transform.penalty(coeffs))
    assert max(penalties) / min(penalties) < 1.5
    # (100 mm)^2 x mean squared second derivatives: analytic ~3.5, discretization (padding) included
    assert 2 < penalties[1] < 8


# MIND

def test_mind_descriptors():
    fixed, moving, _, _ = textured_pair((32, 32))
    reference = ex.mind_descriptors(fixed, (0, 1))
    assert reference.shape == (4, 32, 32)
    assert np.allclose(reference.max(axis=0), 1)
    # invariant to linear intensity mappings
    assert np.allclose(ex.mind_descriptors(3 * fixed + 2, (0, 1)), reference)
    # other image: different descriptors
    assert np.abs(ex.mind_descriptors(moving, (0, 1)) - reference).mean() > 0.05


def test_mind_axes():
    """thick axes (in either image) are excluded from the offsets"""
    metric = ex.MetricMIND()
    thin = ex.Image(np.zeros((8, 8, 8)), spacing=(1, 1, 1))
    thick = ex.Image(np.zeros((8, 8, 8)), spacing=(1, 1, 5))
    assert metric.axes(thin) == (0, 1, 2)
    assert metric.axes(thin, thick) == (0, 1)


@pytest.mark.parametrize("transform, params", [("translation", [0.3, -0.2]), ("rigid", [0.02, 0.3, -0.2])])
def test_gradient_mind(transform, params):
    fixed, moving, _, _ = textured_pair((32, 32))
    config = {"transform": transform, "metric": "mind", "interpolator": "bspline"}
    objective = make_objective(fixed, np.exp(-1.5 * moving), config)
    _, gradient = objective(params, gradient=True)
    assert np.allclose(gradient, finite_gradient(objective, params), rtol=1e-3, atol=1e-9)


# folding check (Jacobian determinant)

def test_jacobian_global():
    geometry = ex.Geometry((6, 5, 4), spacing=(1, 2, 3))
    A = np.array([[1.1, 0.1, 0.0], [0.0, 0.9, 0.2], [0.05, 0.0, 1.2]])
    affine = ex.Transformation("affine", geometry, np.r_[A.ravel(), [1.0, 2.0, 3.0]])
    assert np.allclose(ex.spatial_jacobian(affine)[0], affine.affine()[0])
    det = ex.jacobian_determinant(affine)
    assert isinstance(det, io.Volume) and det.shape == (6, 5, 4)
    assert np.allclose(det, np.linalg.det(A))
    # rigid: volume preserving; sequence: product
    rigid = rigid_transformation(geometry)
    assert np.allclose(ex.jacobian_determinant(rigid), 1)
    sequence = ex.TransformationSequence([rigid, affine])
    assert np.allclose(ex.jacobian_determinant(sequence), np.linalg.det(A))
    assert ex.check_folding(sequence).ok


def test_jacobian_bspline():
    geometry = ex.Geometry((20, 16), spacing=(1.0, 1.5))
    transform = ex.TransformBSpline(grid_spacing=5)
    transform.setup(geometry)
    zero = ex.Transformation(transform, geometry, transform.init(2))
    assert np.allclose(ex.jacobian_determinant(zero), 1)
    # linear displacement u = (0.1 x, -0.2 y): reproduced by cubic B-splines, det = 1.1 x 0.8
    control = transform.control_points()
    params = np.r_[0.1 * control[:, 0], -0.2 * control[:, 1]]
    linear = ex.Transformation(transform, geometry, params)
    det = ex.jacobian_determinant(linear)
    assert isinstance(det, ex.Image) and np.allclose(det, 1.1 * 0.8)
    field = ex.deformation_field(linear)
    assert field.shape == (20, 16, 2)
    assert np.allclose(field.reshape(-1, 2), np.c_[0.1, -0.2] * geometry.points)


def test_check_folding():
    geometry = ex.Geometry((24, 24))
    transform = ex.TransformBSpline(grid_spacing=4)
    transform.setup(geometry)
    # one control point displaced by 20 mm along x: max du/dx = 20 x max|beta3'| x beta3(0) / 4 mm = 1.67, folding
    params = np.zeros(transform.init(2).size)
    params[np.ravel_multi_index((4, 4), tuple(transform.grid["shape"]))] = 20.0
    check = ex.check_folding(ex.Transformation(transform, geometry, params))
    assert not check.ok and check.folded > 0 and check.min <= 0
    assert np.isclose(check.fraction, check.folded / 24**2)
    # small displacements: no folding
    small = ex.check_folding(ex.Transformation(transform, geometry, 0.1 * params))
    assert small.ok and 0 < small.min < 1 < small.max
    # mask: only the masked points
    mask = np.zeros((24, 24), dtype=bool)
    mask[:2, :2] = True
    masked = ex.check_folding(ex.Transformation(transform, geometry, params), mask=mask)
    assert masked.folded <= 4
    # other grid
    ref = ex.Image(np.zeros((12, 12)), spacing=2.0)
    assert ex.check_folding(ex.Transformation(transform, geometry, params), ref=ref).determinant.shape == (12, 12)


# sparse ROI slice interpolation

def disks(shape, centers, radii, label=1):
    """2D disks stacked along the last axis (radius 0: empty slice)"""
    x, y = np.indices(shape[:2])
    out = np.zeros(shape, dtype=np.uint8)
    for k, ((cx, cy), r) in enumerate(zip(centers, radii)):
        out[..., k][(x - cx) ** 2 + (y - cy) ** 2 <= r**2] = label
    return out


def test_interpolate_slices_shape():
    nz = 9
    radii = np.linspace(4, 12, nz)
    truth = disks((40, 40, nz), [(20, 20)] * nz, radii)
    sparse = np.zeros_like(truth)
    sparse[..., [0, 4, 8]] = truth[..., [0, 4, 8]]
    filled = ex.interpolate_slices(sparse)
    assert isinstance(filled, io.Volume) and filled.shape == truth.shape
    assert np.array_equal(np.asarray(filled)[..., [0, 4, 8]], truth[..., [0, 4, 8]])
    # radii interpolated linearly: area within a pixel ring of the truth
    for k in range(nz):
        area = np.count_nonzero(np.asarray(filled)[..., k])
        assert abs(area - np.pi * radii[k] ** 2) < 2 * np.pi * radii[k], k


def test_interpolate_slices_labels_axis():
    # two labels, slices along axis 0, label 2 drawn on one side only: vanishes half-way
    truth = np.zeros((5, 30, 30), dtype=np.uint8)
    truth[0, 5:15, 5:15] = 1
    truth[4, 8:18, 8:18] = 1
    truth[0, 20:28, 20:28] = 2
    filled = np.asarray(ex.interpolate_slices(truth, axis=0, slices=[0, 4]))
    assert set(np.unique(filled[1])) == {0, 1, 2}
    assert 2 not in filled[3] and np.count_nonzero(filled[3] == 1) > 50
    # out of range slices: unchanged; 2D ROI: Image
    out = ex.interpolate_slices(truth[:, :, 10], axis=0)
    assert isinstance(out, ex.Image)
    with pytest.raises(ValueError):
        ex.interpolate_slices(truth, method="registration")


def test_interpolate_slices_registration():
    """a disk moving across the slices: the image guides the interpolation (shape-based fails: no overlap)"""
    nz = 5
    centers = [(12 + 4 * k, 20) for k in range(nz)]
    truth = disks((40, 40, nz), centers, [5] * nz)
    rng = np.random.default_rng(0)
    image = truth * 100.0 + rng.normal(scale=2, size=truth.shape)
    image = ex.gaussian_filter(image, (1, 1, 0))
    sparse = np.zeros_like(truth)
    sparse[..., [0, 4]] = truth[..., [0, 4]]

    def dice(a, b):
        a, b = np.asarray(a) > 0, b > 0
        return 2 * (a & b).sum() / (a.sum() + b.sum())

    config = {"transform": {"type": "bspline", "grid_spacing": 8}, "metric": "mse", "optimizer": {"type": "sgd", "niter": 64}}
    shape = ex.interpolate_slices(sparse)
    registration = ex.interpolate_slices(sparse, image, method="registration", config=config)
    assert dice(np.asarray(shape)[..., 2], truth[..., 2]) < 0.3
    assert dice(np.asarray(registration)[..., 2], truth[..., 2]) > 0.8

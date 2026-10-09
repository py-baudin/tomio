import numpy as np
import pytest

pytest.importorskip("scipy")

from iomed import volume, interp
Volume = volume.Volume


@pytest.mark.parametrize("method", ["nearest", "spline3"])
def test_interpolate_single_slice(method):
    # single-slice source: the slice is used for target positions off the slice center
    src = Volume(np.random.rand(6, 7, 1), spacing=[1, 1, 3])
    tgt = Volume(np.zeros((6, 7, 1)), spacing=[1, 1, 3], origin=[0, 0, 1])
    out = interp.interpolate_like(tgt, src, method=method)
    assert np.allclose(out, src)


def test_interpolate_roi_sparse_slices_downsampling():
    # labels drawn on non-consecutive slices (z = 3, 8, 13)
    roi = Volume(np.zeros((20, 20, 20), dtype=np.uint8))
    roi[4:12, 4:12, 3] = 1
    roi[6:14, 6:14, 8] = 2
    roi[8:16, 8:16, 13] = 3
    # lower resolution target
    ref = Volume(np.zeros((10, 10, 10)), spacing=[2, 2, 2])

    out = interp.interpolate_roi(ref, roi)
    assert out.shape == ref.shape
    assert out.dtype == roi.dtype
    assert out.spacing == ref.spacing
    # all labels kept, each in the nearest target slice (z / 2, rounded half up)
    for label, z in [(1, 2), (2, 4), (3, 7)]:
        x, y, zs = np.nonzero(out == label)
        assert set(zs) == {z}
        # the downsampled square is filled
        assert np.all(out[x.min():x.max() + 1, y.min():y.max() + 1, z] == label)

    # by comparison, nearest-neighbour interpolation misses the slices between target slices
    nearest = interp.interpolate_like(ref, roi, method="nearest")
    assert set(np.unique(nearest)) == {0, 2}


SHIFTED = Volume(np.zeros((10, 10, 10)), origin=(0.5, 0.3, 0))


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int64])
def test_interpolate_default_integer(dtype):
    # label maps: default nearest, no new values
    lab = Volume(np.random.randint(0, 4, (10, 10, 10)).astype(dtype))
    out = interp.interpolate_like(SHIFTED, lab)
    assert out.dtype == dtype
    assert set(np.unique(out)) <= {0, 1, 2, 3}
    assert np.array_equal(out, interp.interpolate_like(SHIFTED, lab, method="nearest"))


def test_interpolate_default_bool():
    mask = Volume(np.random.rand(10, 10, 10) > 0.5)
    out = interp.interpolate_like(SHIFTED, mask)
    assert out.dtype == bool
    assert np.array_equal(out, interp.interpolate_like(SHIFTED, mask.astype(np.uint8), method="nearest"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interpolate_default_float(dtype):
    src = Volume(np.random.rand(10, 10, 10).astype(dtype))
    out = interp.interpolate_like(SHIFTED, src)
    assert out.dtype == dtype
    assert np.array_equal(out, interp.interpolate_like(SHIFTED, src, method="spline3"))


def test_interpolate_explicit_method_integer():
    # an explicit method is used for integer data too, output keeps the input dtype
    lab = Volume(np.random.randint(0, 100, (10, 10, 10)).astype(np.int16))
    out = interp.interpolate_like(SHIFTED, lab, method="spline1")
    assert out.dtype == np.int16
    assert not np.array_equal(out, interp.interpolate_like(SHIFTED, lab, method="nearest"))


def test_interpolate_default_mixed_sources():
    src = Volume(np.random.rand(10, 10, 10))
    lab = Volume(np.random.randint(0, 4, (10, 10, 10)))
    geom = interp.get_geometry(src)
    out_src, out_lab = interp.interpolate(interp.get_geometry(SHIFTED), geom, src, lab)
    assert np.array_equal(out_src, interp.interpolate_like(SHIFTED, src, method="spline3"))
    assert np.array_equal(out_lab, interp.interpolate_like(SHIFTED, lab, method="nearest"))


def test_resample_like():
    src = Volume(np.random.rand(10, 10, 10), spacing=[1, 2, 3])
    out = interp.resample(src, like=SHIFTED)
    assert out.shape == SHIFTED.shape
    assert out.origin == SHIFTED.origin
    assert np.array_equal(out, interp.interpolate_like(SHIFTED, src))


def test_resample_spacing():
    src = Volume(np.random.rand(10, 12, 14), spacing=[1, 2, 3], origin=[4, 5, 6])
    out = interp.resample(src, spacing=(0.5, 1, 1.5))
    assert out.shape == (19, 23, 27)
    assert out.spacing == (0.5, 1.0, 1.5)
    assert np.array_equal(out, interp.interpolate_to({"spacing": (0.5, 1, 1.5)}, src))


def test_resample_geometry():
    src = Volume(np.random.rand(10, 10, 10))
    geometry = {"origin": (1, 1, 1), "spacing": (2, 2, 2), "shape": (3, 4, 5)}
    out = interp.resample(src, geometry=geometry)
    assert out.shape == (3, 4, 5)
    assert out.origin == (1, 1, 1)
    # missing shape: target covers the source
    out = interp.resample(src, geometry={"spacing": (2, 2, 2)})
    assert out.origin == src.origin
    assert out.shape == (5, 5, 5)


def test_resample_methods():
    src = Volume(np.random.rand(10, 10, 10))
    assert np.array_equal(
        interp.resample(src, like=SHIFTED, method="linear"),
        interp.interpolate_like(SHIFTED, src, method="spline1"),
    )
    roi = Volume(np.random.randint(0, 3, (10, 10, 10)).astype(np.uint8))
    ref = Volume(np.zeros((20, 20, 20)), spacing=[0.5, 0.5, 0.5])
    assert np.array_equal(interp.resample(roi, like=ref, method="roi"), interp.interpolate_roi(ref, roi))
    labels = interp.resample(roi, like=ref, method="labels")
    assert np.array_equal(labels, interp.interpolate_labels(ref, roi))
    assert labels.dtype == np.uint8
    assert labels.spacing == ref.spacing
    # spacing only, with roi method
    assert interp.resample(roi, spacing=(0.5, 0.5, 0.5), method="roi").shape == (19, 19, 19)


def test_resample_fill():
    src = Volume(np.random.rand(10, 10, 10))
    out = interp.resample(src, like=Volume(np.zeros((10, 10, 10)), origin=(-2, 0, 0)), method="linear", fill=np.nan)
    assert np.all(np.isnan(out[:2]))
    assert not np.any(np.isnan(out[2:]))
    with pytest.raises(ValueError):
        interp.resample(src.astype(int), like=SHIFTED, fill=np.nan)


def test_resample_errors():
    src = Volume(np.random.rand(10, 10, 10))
    with pytest.raises(ValueError):
        interp.resample(src)
    with pytest.raises(ValueError):
        interp.resample(src, like=src, spacing=(1, 1, 1))
    with pytest.raises(ValueError):
        interp.resample(src, like=src, method="cubic")


def test_interp_geometry_objects():
    # Geometry objects are accepted in place of geometry dicts
    src = Volume(np.random.rand(10, 10, 10), spacing=[1, 2, 3])
    geom = SHIFTED.geometry
    expected = interp.interpolate_like(SHIFTED, src)
    assert np.array_equal(interp.interpolate(geom, src.geometry, src), expected)
    assert np.array_equal(interp.interpolate_to(geom, src), expected)
    assert np.array_equal(interp.resample(src, geometry=geom), expected)
    assert np.array_equal(interp.resample(src, like=geom), expected)
    assert interp.resample(src, like=geom).geometry == geom
    roi = Volume(np.random.randint(0, 3, (10, 10, 10)).astype(np.uint8))
    assert np.array_equal(interp.resample(roi, like=geom, method="labels"), interp.resample(roi, like=SHIFTED, method="labels"))


def test_resample_exports():
    import iomed
    src = Volume(np.random.rand(10, 10, 10))
    expected = interp.resample(src, like=SHIFTED)
    assert np.array_equal(volume.resample(src, like=SHIFTED), expected)
    assert np.array_equal(iomed.resample(src, like=SHIFTED), expected)


def concentric_spheres(vol, center, radii):
    """label map: label i+1 for radius in [radii[i-1], radii[i]["""
    dist = np.linalg.norm(vol.get_coords(np.indices(vol.shape).reshape(3, -1).T) - center, axis=1)
    labels = np.searchsorted(radii, dist, side="right") + 1
    labels[labels > len(radii)] = 0
    return Volume(labels.reshape(vol.shape).astype(np.uint8), **vol.header)


@pytest.mark.parametrize("method", ["spline1", "spline3"])
def test_interpolate_roi_upsampling_quality(method):
    # low resolution label map of concentric spheres, upsampled x2
    center, radii = (11, 11, 11), [3.5, 6.5, 9]
    roi = concentric_spheres(Volume(np.zeros((12, 12, 12)), spacing=[2, 2, 2]), center, radii)
    ref = Volume(np.zeros((23, 23, 23)))
    truth = concentric_spheres(ref, center, radii)

    default = interp.interpolate_roi(ref, roi, disjoint=False)
    smooth = interp.interpolate_roi(ref, roi, method=method, disjoint=False)
    assert smooth.dtype == roi.dtype
    assert smooth.spacing == ref.spacing
    assert set(np.unique(smooth)) == {0, 1, 2, 3}  # no intermediate labels
    # fewer misclassified voxels than the default method
    assert np.sum(smooth != truth) < np.sum(default != truth)


def test_interpolate_roi_upsampling_disjoint():
    # thick slices: with disjoint=True, labels are not interpolated across slices
    rng = np.random.default_rng(0)
    roi = Volume(rng.integers(0, 3, (8, 8, 4)).astype(np.uint8), spacing=[1, 1, 4])
    ref = Volume(np.zeros((8, 8, 13)))
    out = interp.interpolate_roi(ref, roi, method="spline1", axis=2, disjoint=True)
    for z in range(13):
        assert np.array_equal(out[..., z], roi[..., int(np.round(z / 4))])


def test_interpolate_labels_outside():
    # outside the label map: background
    labelmap = Volume(np.ones((5, 5, 5), dtype=np.uint8))
    ref = Volume(np.zeros((9, 9, 9)), origin=[-2, -2, -2])
    out = interp.interpolate_labels(ref, labelmap, method="spline1")
    assert np.all(out[2:7, 2:7, 2:7] == 1)
    assert np.sum(out) == 5**3


def last_voxel(vol):
    return vol.get_coords(np.array(vol.shape) - 1)


def test_interpolate_to_same_spacing():
    src = Volume(np.random.rand(10, 12, 14), spacing=[1, 2, 3], origin=[4, 5, 6])
    out = interp.interpolate_to({"spacing": src.spacing}, src, method="nearest")
    assert out.shape == src.shape
    assert np.allclose(out.origin, src.origin)
    assert np.allclose(out, src)


def test_interpolate_to_half_spacing():
    src = Volume(np.random.rand(10, 12, 14), spacing=[1, 2, 3], origin=[4, 5, 6])
    out = interp.interpolate_to({"spacing": (0.5, 1, 1.5)}, src)
    assert out.shape == (19, 23, 27)
    assert out.spacing == (0.5, 1.0, 1.5)
    # same world extent
    assert np.allclose(out.origin, src.origin)
    assert np.allclose(last_voxel(out), last_voxel(src))


def test_interpolate_to_rotated():
    arr = np.random.rand(4, 5, 6)
    src = Volume(arr)
    # target axis 0 along +y, axis 1 along -x
    tr = ((0, 1, 0), (-1, 0, 0), (0, 0, 1))
    out = interp.interpolate_to({"transform": tr}, src, method="nearest")
    assert out.shape == (5, 4, 6)
    assert np.allclose(out.origin, (3, 0, 0))
    # out[i, j, k] is at world (3 - j, i, k)
    assert np.allclose(out, arr[::-1].transpose(1, 0, 2))


def test_interpolate_keeps_input_nans():
    src = Volume(np.where(np.random.rand(8, 8, 8) > 0.9, np.nan, 1.0))
    isnan = np.isnan(src)
    out = interp.interpolate_like(src, src)
    # input unchanged
    assert np.array_equal(np.isnan(src), isnan)
    # NaNs propagated to the output
    assert np.array_equal(np.isnan(out), isnan)

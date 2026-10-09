import numpy as np
import pytest

pytest.importorskip("scipy")

from iomed import volume, interp
Volume = volume.Volume


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

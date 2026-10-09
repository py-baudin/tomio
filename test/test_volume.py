import pathlib
import numpy as np
import pytest

from iomed import volume
Volume = volume.Volume

def test_volume_class():
    arr = np.arange(3*4*5).reshape(3,4,5)

    vol = Volume(arr)
    assert vol.spacing == (1.0, 1.0, 1.0)
    assert vol.origin == (0.0, 0.0, 0.0)
    assert vol.transform == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    tr = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
    vol = Volume(arr, spacing=[1,2,3], origin=[4,5,6], transform=tr, metadata={'foo': 'bar'})
    assert vol.spacing == (1.0, 2.0, 3.0)
    assert vol.origin == (4.0, 5.0, 6.0)
    assert vol.transform == ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0,  0.0))
    assert vol.metadata == {'foo': 'bar'}

    # view/ufunc
    assert vol[0].spacing == (1.0, 2.0, 3.0)
    assert vol[0].origin == (4.0, 5.0, 6.0)
    assert vol[0].transform == ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0,  0.0))
    assert vol[0].metadata == {'foo': 'bar'}

    assert (vol**2).spacing == (1.0, 2.0, 3.0)
    assert (vol**2).origin == (4.0, 5.0, 6.0)
    assert (vol**2).transform == ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0,  0.0))
    assert (vol**2).metadata == {'foo': 'bar'}

    # affine transform matrix
    assert np.allclose(vol.affine, volume.make_affine_transform(vol.origin, vol.spacing, vol.transform))

    # anatomical orientation
    assert vol.anatomical_orientation == 'RIA'
    assert vol.reorient('RAI').transform == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0,  1.0))



def test_volume_utils():
    arr = np.arange(3*4*5).reshape(3,4,5)
    vol = Volume(arr, spacing=[1,2,3], origin=[4,5,6], metadata={'foo': 'bar'})

    mask = (vol > 4) & (vol < 30)
    values = arr[mask]

    # to volume
    vol2 = volume.tovolume(values, mask)
    assert np.allclose(vol2[mask], values)
    assert vol2.spacing == vol.spacing
    assert vol2.origin == vol.origin
    assert vol2.transform == vol.transform
    assert vol.metadata == vol2.metadata

    vol2 = volume.tovolume(values, mask.A, ref=vol)
    assert np.allclose(vol2[mask], values)
    assert vol2.spacing == vol.spacing
    assert vol2.origin == vol.origin
    assert vol2.transform == vol.transform
    assert vol.metadata == vol2.metadata


def test_volume_io(tmpdir):
    tmp = pathlib.Path(tmpdir)

    arr = np.arange(3*4*5).reshape(3,4,5)
    vol = Volume(arr, spacing=[1,2,3], origin=[4,5,6])

    # metaimage
    volume.write(tmp / 'vol.mha', vol)
    assert (tmp / 'vol.mha').is_file()

    vol2 = volume.read(tmp / 'vol.mha')
    assert np.allclose(vol2, vol)
    assert vol2.spacing == vol.spacing
    assert vol2.origin == vol.origin
    assert vol2.transform == vol.transform

    volume.write(tmp / 'vol.nii.gz', vol)
    assert (tmp / 'vol.nii.gz').is_file()

    vol2 = volume.read(tmp / 'vol.nii.gz')
    assert np.allclose(vol2, vol)
    assert vol2.spacing == vol.spacing
    assert vol2.origin == vol.origin
    assert vol2.transform == vol.transform

    # default extension
    volume.write(tmp / 'vol', vol)
    assert ((tmp / 'vol').with_suffix(volume.DEFAULT_EXT)).is_file()


def test_volume_write_nan_as(monkeypatch, tmp_path):
    written = {}
    def write_volume(filename, array, **kwargs):
        written.update(kwargs, array=np.array(array))
    monkeypatch.setattr(volume.iovolume, 'write_volume', write_volume)

    arr = np.array([[[np.nan, 1.0]]])
    vol = Volume(arr, spacing=[1, 2, 3])
    volume.write(tmp_path / 'vol.mha', vol, nan_as=-1)

    # input unchanged
    assert np.isnan(arr[0, 0, 0])
    assert np.isnan(vol[0, 0, 0])
    # NaNs replaced in written data, geometry kept
    assert np.array_equal(written['array'], [[[-1.0, 1.0]]])
    assert written['spacing'] == (1.0, 2.0, 3.0)







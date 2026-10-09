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


def rotation(ax, ay, az):
    """rotation matrix from angles (degrees) around x, y, z"""
    ax, ay, az = np.radians([ax, ay, az])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


DIRECTIONS = {
    # ITK direction matrices D (columns: world directions of the array axes)
    "permutation": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    "oblique": rotation(10, 20, 30),
    "flip": np.diag([1, -1, 1]),  # improper (det < 0)
}
INDICES = [(0, 0, 0), (1, 2, 3), (2, 3, 4)]


@pytest.mark.parametrize("ext", [".mha", ".nii.gz"])
@pytest.mark.parametrize("direction", list(DIRECTIONS))
def test_volume_io_direction(tmp_path, ext, direction):
    import SimpleITK as sitk

    D = DIRECTIONS[direction]
    arr = np.random.rand(3, 4, 5).astype(np.float32)
    vol = Volume(arr, spacing=[1, 2, 3], origin=[4, 5, 6], transform=D.T)
    filename = str(tmp_path / f"vol{ext}")
    volume.write(filename, vol)

    # check the written file with SimpleITK
    im = sitk.ReadImage(filename)
    assert np.allclose(np.reshape(im.GetDirection(), (3, 3)), D, atol=1e-5)
    for idx in INDICES:
        assert np.allclose(im.TransformIndexToPhysicalPoint(idx), vol.get_coords(idx), atol=1e-4)
        assert im.GetPixel(idx) == vol[idx]

    # read back
    vol2 = volume.read(filename)
    assert np.allclose(vol2.transform, vol.transform, atol=1e-5)
    assert np.allclose(vol2.affine, vol.affine, atol=1e-4)
    for idx in INDICES:
        assert np.allclose(vol2.get_coords(idx), vol.get_coords(idx), atol=1e-4)


@pytest.mark.parametrize("ext", [".mha", ".nii.gz"])
def test_volume_read_direction(tmp_path, ext):
    """read a file written directly with SimpleITK"""
    import SimpleITK as sitk

    D = DIRECTIONS["oblique"]
    im = sitk.GetImageFromArray(np.random.rand(5, 4, 3))  # ITK size: (3, 4, 5)
    im.SetSpacing((1, 2, 3))
    im.SetOrigin((4, 5, 6))
    im.SetDirection(D.ravel())
    filename = str(tmp_path / f"vol{ext}")
    sitk.WriteImage(im, filename)
    im = sitk.ReadImage(filename)

    vol = volume.read(filename)
    assert vol.shape == im.GetSize()
    for idx in INDICES:
        assert np.allclose(vol.get_coords(idx), im.TransformIndexToPhysicalPoint(idx), atol=1e-4)
        assert vol[idx] == im.GetPixel(idx)


GEOMETRY = dict(spacing=[1, 2, 3], origin=[4, 5, 6], transform=DIRECTIONS["oblique"].T)


@pytest.mark.parametrize("ext", [".mha", ".nii.gz"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(3, 4, 5), (3, 4, 5, 6)])
def test_volume_io_complex(tmp_path, ext, dtype, shape):
    import SimpleITK as sitk

    arr = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(dtype)
    vol = Volume(arr, **GEOMETRY)
    filename = str(tmp_path / f"vol{ext}")
    volume.write(filename, vol)

    # stored as a 2-component vector image
    im = sitk.ReadImage(filename)
    assert im.GetNumberOfComponentsPerPixel() == 2
    assert im.GetSize() == shape
    idx = (1, 2, 3) + (0,) * (len(shape) - 3)
    assert np.allclose(im.GetPixel(idx), (arr[idx].real, arr[idx].imag))

    vol2 = volume.read(filename, as_complex=True)
    assert vol2.dtype == dtype
    assert np.array_equal(vol2, arr)
    assert np.allclose(vol2.affine, vol.affine, atol=1e-4)

    # without as_complex: components in the last axis
    vol3 = volume.read(filename)
    assert vol3.shape == shape + (2,)
    assert np.array_equal(vol3[..., 1], arr.imag)


@pytest.mark.parametrize("ext", [".mha", ".nii.gz"])
def test_volume_io_4d(tmp_path, ext):
    import SimpleITK as sitk

    arr = np.random.rand(3, 4, 5, 6).astype(np.float32)
    vol = Volume(arr, **GEOMETRY)
    filename = str(tmp_path / f"vol{ext}")
    volume.write(filename, vol)

    im = sitk.ReadImage(filename)
    assert im.GetSize() == (3, 4, 5, 6)
    assert im.GetSpacing()[3] == 1
    for idx in INDICES:
        assert np.allclose(im.TransformIndexToPhysicalPoint(idx + (0,))[:3], vol.get_coords(idx), atol=1e-4)

    vol2 = volume.read(filename)
    assert np.array_equal(vol2, arr)
    assert np.allclose(vol2.spacing, vol.spacing)
    assert np.allclose(vol2.affine, vol.affine, atol=1e-4)


def test_volume_read_header(tmp_path):
    vol = Volume(np.zeros((3, 4, 5)), metadata={"PatientName": "foo"}, **GEOMETRY)
    volume.write(tmp_path / "vol.mha", vol)
    header = volume.read_header(tmp_path / "vol.mha")
    assert header["shape"] == (3, 4, 5)
    assert np.allclose(header["spacing"], vol.spacing)
    assert np.allclose(header["origin"], vol.origin)
    assert np.allclose(header["transform"], vol.transform)
    assert header["metadata"]["PatientName"] == "foo"
    # same header as read
    vol2 = volume.read(tmp_path / "vol.mha")
    assert np.allclose(Volume(vol2, **{k: v for k, v in header.items() if k != "shape"}).affine, vol2.affine)

    # vector image: components in the last axis
    volume.write(tmp_path / "complex.mha", vol + 1j)
    assert volume.read_header(tmp_path / "complex.mha")["shape"] == (3, 4, 5, 2)


def test_volume_read_dtype(tmp_path):
    vol = Volume(np.random.rand(3, 4, 5))
    volume.write(tmp_path / "vol.nii.gz", vol)
    assert volume.read(tmp_path / "vol.nii.gz").dtype == np.float64
    vol2 = volume.read(tmp_path / "vol.nii.gz", dtype=np.float32)
    assert vol2.dtype == np.float32
    assert np.allclose(vol2, vol)


def test_volume_write_compression_level(tmp_path):
    arr = np.tile(np.arange(20, dtype=np.int16), (20, 20, 1))  # compressible
    sizes = {}
    for level in [1, 9]:
        filename = tmp_path / f"vol{level}.nii.gz"
        volume.write(filename, arr, compression_level=level)
        assert np.array_equal(volume.read(filename), arr)
        sizes[level] = filename.stat().st_size
    assert sizes[9] <= sizes[1]


def test_volume_io_metadata(tmp_path):
    vol = Volume(np.zeros((3, 4, 5)), metadata={"PatientName": "foo", "Count": 3})
    volume.write(tmp_path / "vol.mha", vol)
    vol2 = volume.read(tmp_path / "vol.mha")
    # MetaImage keeps arbitrary keys (values are strings)
    assert vol2.metadata["PatientName"] == "foo"
    assert vol2.metadata["Count"] == "3"


def test_volume_io_nifti_rewrite(tmp_path):
    """volume read from NIfTI (with NIfTI-specific metadata) can be written again unchanged"""
    arr = np.random.rand(3, 4, 5).astype(np.float32)
    vol = Volume(arr, spacing=[1, 2, 3], origin=[4, 5, 6], transform=DIRECTIONS["oblique"].T)
    volume.write(tmp_path / "vol1.nii.gz", vol)
    vol1 = volume.read(tmp_path / "vol1.nii.gz")
    assert vol1.metadata  # NIfTI header fields

    volume.write(tmp_path / "vol2.nii.gz", vol1)
    vol2 = volume.read(tmp_path / "vol2.nii.gz")
    assert np.array_equal(vol2, arr)
    assert np.allclose(vol2.affine, vol.affine, atol=1e-4)


def test_volume_write_nan_as(monkeypatch, tmp_path):
    written = {}
    def write_image(filename, array, **kwargs):
        written.update(kwargs, array=np.array(array))
    monkeypatch.setattr(volume.imagefile, 'write_image', write_image)

    arr = np.array([[[np.nan, 1.0]]])
    vol = Volume(arr, spacing=[1, 2, 3])
    volume.write(tmp_path / 'vol.mha', vol, nan_as=-1)

    # input unchanged
    assert np.isnan(arr[0, 0, 0])
    assert np.isnan(vol[0, 0, 0])
    # NaNs replaced in written data, geometry kept
    assert np.array_equal(written['array'], [[[-1.0, 1.0]]])
    assert written['spacing'] == (1.0, 2.0, 3.0)







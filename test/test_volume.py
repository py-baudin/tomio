import pathlib
import dataclasses
import numpy as np
import pytest

from tomio import volume
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
    assert vol[:1].spacing == (1.0, 2.0, 3.0)
    assert vol[:1].origin == (4.0, 5.0, 6.0)
    assert vol[:1].transform == ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0,  0.0))
    assert vol[:1].metadata == {'foo': 'bar'}
    assert type(vol[0]) is np.ndarray  # integer index: no geometry

    assert (vol**2).spacing == (1.0, 2.0, 3.0)
    assert (vol**2).origin == (4.0, 5.0, 6.0)
    assert (vol**2).transform == ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0,  0.0))
    assert (vol**2).metadata == {'foo': 'bar'}

    # affine transform matrix
    assert np.allclose(vol.affine, volume.make_affine_transform(vol.origin, vol.spacing, vol.transform))

    # anatomical orientation
    assert vol.anatomical_orientation == 'RIA'
    assert vol.reorient('RAI').transform == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0,  1.0))



def check_slice(vol, key):
    """the sliced volume has the data of the slice, and its voxels keep their world coordinates"""
    out = vol[key]
    assert isinstance(out, Volume)
    assert np.array_equal(out.A, vol.A[key])
    assert out.metadata == vol.metadata
    if out.size == 0:
        return out
    # original indices of each voxel of the slice
    key = key if isinstance(key, tuple) else (key,)
    orig = np.indices(vol.shape)[(slice(None),) + key].reshape(vol.ndim, -1).T
    new = np.indices(out.shape).reshape(out.ndim, -1).T
    assert np.allclose(out.get_coords(new[:, :3]), vol.get_coords(orig[:, :3]))
    return out


SLICES = [
    np.s_[2:],
    np.s_[::2],
    np.s_[:, 5:15, ::3],
    np.s_[::-1],
    np.s_[3:8:2, ::-2, -1:0:-3],
    np.s_[..., ::-1],
    np.s_[..., 2:4],
    np.s_[8:2],  # empty
]


@pytest.mark.parametrize("key", SLICES)
def test_volume_slicing(key):
    tr = np.array([[0.8, 0.6, 0], [-0.6, 0.8, 0], [0, 0, 1]])
    vol = Volume(np.random.rand(10, 12, 14), spacing=[1, 2, 3], origin=[4, 5, 6], transform=tr, metadata={"a": "b"})
    check_slice(vol, key)


def test_volume_slicing_values():
    vol = Volume(np.zeros((10, 20, 30)), spacing=[1, 2, 3])
    assert vol[2:].origin == (2, 0, 0)
    assert vol[::2].spacing == (2, 2, 3)
    out = vol[:, 5:15, ::3]
    assert out.origin == (0, 10, 0)
    assert out.spacing == (1, 2, 9)
    out = vol[::-1]
    assert out.origin == (9, 0, 0)
    assert out.transform == ((-1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert out.anatomical_orientation == "LAI"


@pytest.mark.parametrize("key", [np.s_[0], np.s_[:, 3], np.s_[..., 2], np.s_[[1, 2]], np.s_[None], np.s_[:, None]])
def test_volume_indexing_plain_array(key):
    vol = Volume(np.random.rand(10, 12, 14), spacing=[1, 2, 3])
    out = vol[key]
    assert type(out) is np.ndarray
    assert np.array_equal(out, vol.A[key])


def test_volume_indexing_mask_scalar():
    vol = Volume(np.random.rand(10, 12, 14))
    assert type(vol[vol > 0.5]) is np.ndarray
    assert not isinstance(vol[1, 2, 3], Volume)
    assert vol[1, 2, 3] == vol.A[1, 2, 3]


@pytest.mark.parametrize("key", [np.s_[..., 0], np.s_[1:, ..., 2], np.s_[::-1, :, 2:, 1:3]])
def test_volume_slicing_4d(key):
    # integer index on non-spatial axes: geometry unchanged
    vol = Volume(np.random.rand(4, 5, 6, 7), spacing=[1, 2, 3], origin=[4, 5, 6])
    check_slice(vol, key)


def test_volume_flip():
    vol = Volume(np.random.rand(4, 5, 6), spacing=[1, 2, 3], origin=[4, 5, 6])
    flipped = np.flip(vol, axis=(0, 2))
    assert isinstance(flipped, Volume)
    assert np.allclose(flipped.get_coords([0, 0, 0]), vol.get_coords([3, 0, 5]))

    # reorient: each voxel keeps its value at the same world position
    for orientation in ["LAI", "RPS", "AIR", "SPL"]:
        out = vol.reorient(orientation)
        assert out.anatomical_orientation == orientation
        new = np.indices(out.shape).reshape(3, -1).T
        orig = np.round(vol.get_indices(out.get_coords(new))).astype(int)
        assert np.array_equal(out.A[tuple(new.T)], vol.A[tuple(orig.T)])


def oblique_volume(shape):
    tr = np.array([[0.8, 0.6, 0], [-0.6, 0.8, 0], [0, 0, 1]])
    return Volume(np.random.rand(*shape), spacing=[1, 2, 3], origin=[4, 5, 6], transform=tr, metadata={"a": "b"})


@pytest.mark.parametrize(
    "func",
    [
        lambda v: v.sum(axis=0),
        lambda v: v.mean(axis=(0, 1)),
        lambda v: v.max(axis=2, keepdims=True),
        lambda v: np.std(v, axis=1),
        lambda v: np.add.reduce(v),  # default axis: 0
        lambda v: np.median(v, axis=0),
        # not ufuncs
        lambda v: v.argmax(axis=0),
        lambda v: np.argmin(v, axis=2),
        lambda v: v.argmax(axis=1, keepdims=True),
        lambda v: np.nanargmax(v, axis=0),
        lambda v: np.percentile(v, 50, axis=0),
    ],
)
def test_volume_reduction_spatial(func):
    vol = oblique_volume((4, 5, 6))
    out = func(vol)
    assert type(out) is np.ndarray
    assert np.allclose(out, func(vol.A))


def test_volume_reduction_all():
    vol = oblique_volume((4, 5, 6))
    for out in [vol.sum(), vol.mean(), np.max(vol), vol.std(), vol.argmax(), np.argmin(vol)]:
        assert not isinstance(out, Volume)
        assert np.ndim(out) == 0


@pytest.mark.parametrize(
    "func",
    [
        lambda v: v.mean(axis=3),
        lambda v: v.sum(axis=-1),
        lambda v: np.max(v, axis=3),
        lambda v: v.argmax(axis=3),
        lambda v: np.argmin(v, axis=-1),
        lambda v: np.var(v, axis=3),
    ],
)
def test_volume_reduction_non_spatial(func):
    # reduction over non-spatial axes: geometry kept
    vol = oblique_volume((4, 5, 6, 7))
    out = func(vol)
    assert isinstance(out, Volume)
    assert out.shape == (4, 5, 6)
    assert out.header == vol.header
    assert np.allclose(out, func(vol.A))
    assert type(vol.sum(axis=(2, 3))) is np.ndarray


def test_volume_argmax_out():
    vol = oblique_volume((4, 5, 6))
    out = np.empty((5, 6), dtype=np.intp)
    assert vol.argmax(axis=0, out=out) is out
    assert np.array_equal(out, vol.A.argmax(axis=0))


def test_volume_ufunc_header():
    vol = oblique_volume((4, 5, 6))
    other = Volume(np.ones((4, 5, 6)))
    # elementwise: header of the first Volume input with the shape of the result
    assert (vol + 1).header == vol.header
    assert (1 + vol).header == vol.header
    assert (vol[:1] + vol).header == vol.header  # broadcast: header of `vol`
    assert (vol + other).header == vol.header
    assert (other + vol).header == other.header
    assert np.cumsum(vol, axis=0).header == vol.header
    q, r = np.divmod(vol, 0.3)
    assert q.header == r.header == vol.header
    # shape changed: plain array
    assert type(np.matmul(vol, np.ones((6, 2)))) is np.ndarray
    assert type(np.multiply.outer(vol, [1, 2])) is np.ndarray
    # out argument: modified in place
    out = np.add(vol, 1, out=vol)
    assert out is vol
    assert out.origin == (4, 5, 6)


def check_permutation(vol, func):
    """the permuted volume has permuted data, and its voxels keep their world coordinates"""
    out = func(vol)
    assert isinstance(out, Volume)
    assert np.array_equal(out.A, func(vol.A))
    # original indices of each voxel
    orig = np.stack([func(idx) for idx in np.indices(vol.shape)]).reshape(vol.ndim, -1).T
    new = np.indices(out.shape).reshape(out.ndim, -1).T
    assert np.allclose(out.get_coords(new[:, :3]), vol.get_coords(orig[:, :3]))


PERMUTATIONS = [
    lambda v: v.T,
    lambda v: v.transpose(),
    lambda v: v.transpose(1, 0, 2),
    lambda v: v.transpose((2, 0, 1)),
    lambda v: np.transpose(v, (2, 0, 1)),
    lambda v: np.moveaxis(v, 0, -1),
    lambda v: np.swapaxes(v, 0, 2),
    lambda v: v.swapaxes(-1, 1),
    lambda v: np.rollaxis(v, 2),
    lambda v: v.T[::-1, 2:],  # with slicing
]


@pytest.mark.parametrize("func", PERMUTATIONS)
def test_volume_permutation(func):
    check_permutation(oblique_volume((4, 5, 6)), func)


def test_volume_permutation_4d():
    vol = oblique_volume((4, 5, 6, 7))
    check_permutation(vol, lambda v: v.transpose(1, 0, 2, 3))
    check_permutation(vol, lambda v: np.swapaxes(v, 0, 2))
    # spatial axis moved beyond the first 3 axes: plain array
    for func in [lambda v: v.T, lambda v: np.moveaxis(v, 3, 0), lambda v: v.swapaxes(2, 3)]:
        assert type(func(vol)) is np.ndarray


def test_geometry_class():
    geom = volume.Geometry(shape=(3, 5, 7))
    assert geom.spacing == (1.0, 1.0, 1.0)
    assert geom.origin == (0.0, 0.0, 0.0)
    assert geom.transform == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert np.allclose(geom.center, (1, 2, 3))
    assert geom.extent == (3.0, 5.0, 7.0)
    assert geom.voxel_volume == 1.0

    # normalized values
    geom = volume.Geometry(shape=np.array([3, 5, 7]), spacing=[1, 2, 3], origin=np.zeros(3), transform=np.diag([2, 1, 1]))
    assert geom.shape == (3, 5, 7) and type(geom.shape[0]) is int
    assert geom.spacing == (1.0, 2.0, 3.0)
    assert geom.transform == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert geom.extent == (3.0, 10.0, 21.0)
    assert geom.voxel_volume == 6.0
    assert np.allclose(geom.center, (1, 4, 9))

    # immutable, hashable
    with pytest.raises(dataclasses.FrozenInstanceError):
        geom.spacing = (1, 1, 1)
    assert geom == volume.Geometry((3, 5, 7), (1, 2, 3))
    assert len({geom, volume.Geometry((3, 5, 7), (1, 2, 3))}) == 1

    # invalid values
    for kwargs in [dict(shape=(3, 5)), dict(shape=(3, 5, 7), spacing=(1, 2)), dict(shape=(3, 5, 7), transform=np.eye(2))]:
        with pytest.raises(ValueError):
            volume.Geometry(**kwargs)

    # dict
    assert volume.Geometry.from_dict(geom.to_dict()) == geom
    assert volume.Geometry.from_dict({"shape": (3, 5, 7), "spacing": (1, 2, 3)}) == geom


def test_volume_geometry():
    vol = oblique_volume((4, 5, 6))
    geom = vol.geometry
    assert geom.shape == vol.shape
    assert (geom.spacing, geom.origin, geom.transform) == (vol.spacing, vol.origin, vol.transform)
    assert np.allclose(geom.affine, vol.affine)
    idx = [(0, 0, 0), (1, 2, 3), (3, 4, 5)]
    assert np.allclose(geom.get_coords(idx), vol.get_coords(idx))
    assert np.allclose(geom.get_indices(vol.get_coords(idx)), idx)
    assert np.allclose(geom.center, vol.get_coords([1.5, 2, 2.5]))

    # 4d: geometry of the first 3 axes
    assert Volume(np.zeros((4, 5, 6, 7)), **vol.header).geometry == geom

    # create volume from geometry
    vol2 = Volume(np.zeros((4, 5, 6)), geometry=geom)
    assert vol2.geometry == geom
    assert Volume(np.zeros((4, 5, 6, 2)), geometry=geom).geometry == geom
    assert Volume(np.zeros((4, 5, 6)), geometry=geom, spacing=(1, 1, 1)).spacing == (1, 1, 1)  # override
    assert Volume(np.zeros((4, 5, 6)), geometry=geom.to_dict()).geometry == geom
    with pytest.raises(ValueError):
        Volume(np.zeros((4, 5, 5)), geometry=geom)


def test_geometry_same_as():
    vol = oblique_volume((4, 5, 6))
    geom = vol.geometry
    assert geom.same_as(vol)
    assert geom.same_as(geom.to_dict())
    assert geom.same_as(dataclasses.replace(geom, origin=np.add(geom.origin, 1e-5)))
    assert not geom.same_as(dataclasses.replace(geom, origin=np.add(geom.origin, 1e-3)))
    assert geom.same_as(dataclasses.replace(geom, origin=np.add(geom.origin, 1e-3)), tol=1e-2)
    assert not geom.same_as(dataclasses.replace(geom, shape=(4, 5, 7)))
    assert not geom.same_as(vol[::-1])


def test_orientation_convention():
    vol = Volume(np.random.rand(4, 5, 6))
    assert vol.anatomical_orientation == "RAI"
    assert vol.get_orientation() == "RAI"
    assert vol.get_orientation(convention="to") == "LPS"

    for code in ["LPS", "RAS", "ASL", "IRP"]:
        out = vol.reorient(code, convention="to")
        assert out.get_orientation(convention="to") == code
        # same as the opposite code in the "from" convention
        assert np.array_equal(out, vol.reorient(volume.convert_orientation(code, "to")))

    with pytest.raises(ValueError):
        vol.get_orientation(convention="towards")
    with pytest.raises(ValueError):
        vol.reorient("RAI", convention="towards")


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


def test_volume_sitk():
    import SimpleITK as sitk

    vol = Volume(np.random.rand(3, 4, 5).astype(np.float32), metadata={"foo": "bar"}, **GEOMETRY)
    im = vol.to_sitk()
    assert isinstance(im, sitk.Image)
    assert im.GetSize() == vol.shape
    assert im.GetMetaData("foo") == "bar"
    for idx in INDICES:
        assert np.allclose(im.TransformIndexToPhysicalPoint(idx), vol.get_coords(idx))
        assert im.GetPixel(idx) == vol[idx]

    vol2 = Volume.from_sitk(im)
    assert np.array_equal(vol2, vol)
    assert vol2.geometry.same_as(vol, tol=1e-8)
    assert vol2.metadata["foo"] == "bar"

    # complex
    cvol = vol + 1j * vol
    im = cvol.to_sitk()
    assert im.GetNumberOfComponentsPerPixel() == 2
    assert np.array_equal(Volume.from_sitk(im, as_complex=True), cvol)


def nibabel_volume(direction):
    return Volume(np.random.rand(3, 4, 5).astype(np.float32), spacing=[1, 2, 3], origin=[4, 5, 6], transform=DIRECTIONS[direction].T)


@pytest.mark.parametrize("direction", list(DIRECTIONS))
def test_volume_nibabel(direction):
    nib = pytest.importorskip("nibabel")
    vol = nibabel_volume(direction)
    img = vol.to_nibabel()
    assert isinstance(img, nib.Nifti1Image)
    assert np.array_equal(np.asanyarray(img.dataobj), vol)
    # RAS coordinates: LPS with x and y flipped
    for idx in INDICES:
        ras = (img.affine @ [*idx, 1])[:3]
        assert np.allclose(ras * [-1, -1, 1], vol.get_coords(idx))

    vol2 = Volume.from_nibabel(img)
    assert np.array_equal(vol2, vol)
    assert vol2.geometry.same_as(vol, tol=1e-8)

    # bool: stored as uint8
    assert np.asanyarray((vol > 0.5).to_nibabel().dataobj).dtype == np.uint8


@pytest.mark.parametrize("direction", list(DIRECTIONS))
def test_volume_nibabel_canonical(direction):
    # nibabel's RAS+ canonical orientation == reorient to "RAS" ("to" convention)
    nib = pytest.importorskip("nibabel")
    vol = nibabel_volume(direction)
    canonical = nib.as_closest_canonical(vol.to_nibabel())
    expected = vol.reorient("RAS", convention="to")
    assert np.array_equal(np.asanyarray(canonical.dataobj), expected)
    assert Volume.from_nibabel(canonical).geometry.same_as(expected, tol=1e-6)


@pytest.mark.parametrize("direction", list(DIRECTIONS))
def test_volume_nibabel_files(tmp_path, direction):
    # files written by tomio (SimpleITK) and read by nibabel, and vice versa
    nib = pytest.importorskip("nibabel")
    vol = nibabel_volume(direction)

    volume.write(tmp_path / "tomio.nii.gz", vol)
    vol2 = Volume.from_nibabel(nib.load(tmp_path / "tomio.nii.gz"))
    assert np.array_equal(vol2, vol)
    assert vol2.geometry.same_as(vol, tol=1e-4)

    nib.save(vol.to_nibabel(), tmp_path / "nibabel.nii.gz")
    vol3 = volume.read(tmp_path / "nibabel.nii.gz")
    assert np.array_equal(vol3, vol)
    assert vol3.geometry.same_as(vol, tol=1e-4)


def test_orientation_oblique():
    # an oblique direction: two axes closest to the same anatomical axis, each assigned to a distinct one
    a, b = np.radians(45), np.radians(40)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    vol = Volume(np.zeros((2, 2, 2)), transform=ry @ rx)
    # the dominant anatomical axis of each axis alone: R, A, L (two left-right axes)
    assert vol.anatomical_orientation == "RAI"

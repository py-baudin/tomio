import SimpleITK as sitk
import numpy as np


def write_volume(filename, array, *, spacing=None, origin=None, transform=None, metadata=None, use_compression=True, is_vector=False):
    """write array (x, y, z, ...) to file

    is_vector: the last axis of `array` contains the components of a vector image
    spacing, origin, transform: 3d geometry (identity for the extra dimensions)
    """
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.bool_):
        array = array.astype('uint8')
    if is_vector:
        # numpy order: reversed image axes, then components
        n = array.ndim - 1
        im = sitk.GetImageFromArray(np.transpose(array, (*reversed(range(n)), n)), isVector=True)
    else:
        # isVector=False: otherwise SimpleITK may guess a vector image from the shape
        im = sitk.GetImageFromArray(array.T, isVector=False)
    ndim = im.GetDimension()
    if spacing is not None:
        im.SetSpacing(pad(spacing, ndim, 1))
    if origin is not None:
        im.SetOrigin(pad(origin, ndim, 0))
    if transform is not None:
        # transform: columns of the direction matrix D
        direction = np.eye(max(ndim, 3))
        direction[:3, :3] = np.asarray(transform).reshape(3, 3).T
        im.SetDirection(direction[:ndim, :ndim].ravel())
    if metadata is not None:
        for key, value in metadata.items():
            im.SetMetaData(str(key), str(value))
    sitk.WriteImage(im, filename, useCompression=use_compression)


def read_image(filename):
    """read file, return array (x, y, z, ...) and 3d geometry

    vector images: components are in the last axis
    """
    im = sitk.ReadImage(filename)
    ndim = im.GetDimension()

    array = sitk.GetArrayFromImage(im)
    if array.ndim > ndim:
        # vector image: (reversed image axes, components)
        array = np.transpose(array, (*reversed(range(ndim)), ndim))
    else:
        array = array.T

    # 3d geometry
    spacing = im.GetSpacing()[:3]
    origin = im.GetOrigin()[:3]
    # GetDirection: direction matrix D, row-major; transform: columns of D
    transform = np.reshape(im.GetDirection(), (ndim, ndim))[:3, :3].T
    metadata = {key: im.GetMetaData(key) for key in im.GetMetaDataKeys()}

    header = dict(spacing=spacing, origin=origin, transform=transform, metadata=metadata)
    return array, header


def pad(values, ndim, fill):
    """pad 3d values for images with more than 3 dimensions"""
    values = list(values)
    return values + [fill] * (ndim - len(values))

""" image file i/o with SimpleITK

arrays are in ITK index order (x, y, z, ...); vector images have the components in the last axis
"""
import SimpleITK as sitk
import numpy as np


def write_image(filename, array, *, spacing=None, origin=None, transform=None, metadata=None, use_compression=True, compression_level=None, is_vector=False):
    """write array (x, y, z, ...) to image file

    is_vector: the last axis of `array` contains the components of a vector image
    spacing, origin, transform: 3d geometry (identity for the extra dimensions)
    compression_level: if set (format-dependent, e.g. 1-9 for gzip), implies use_compression
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
    if compression_level is None:
        sitk.WriteImage(im, str(filename), useCompression=use_compression)
    else:
        sitk.WriteImage(im, str(filename), useCompression=True, compressionLevel=compression_level)


def read_image(filename):
    """read image file, return array (x, y, z, ...) and header (3d geometry and metadata)

    vector images: components are in the last axis
    """
    im = sitk.ReadImage(str(filename))
    ndim = im.GetDimension()

    array = sitk.GetArrayFromImage(im)
    if array.ndim > ndim:
        # vector image: (reversed image axes, components)
        array = np.transpose(array, (*reversed(range(ndim)), ndim))
    else:
        array = array.T

    return array, get_header(im)


def read_header(filename):
    """read image file header only (no pixel data)

    returns: shape (x, y, z, ..., [components]), 3d geometry and metadata
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(filename))
    reader.ReadImageInformation()

    shape = tuple(reader.GetSize())
    if reader.GetNumberOfComponents() > 1:
        shape += (reader.GetNumberOfComponents(),)
    return dict(shape=shape, **get_header(reader))


def get_header(src):
    """3d geometry and metadata of a sitk Image or ImageFileReader"""
    ndim = src.GetDimension()
    spacing = src.GetSpacing()[:3]
    origin = src.GetOrigin()[:3]
    # GetDirection: direction matrix D, row-major; transform: columns of D
    transform = np.reshape(src.GetDirection(), (ndim, ndim))[:3, :3].T
    metadata = {key: src.GetMetaData(key) for key in src.GetMetaDataKeys()}
    return dict(spacing=spacing, origin=origin, transform=transform, metadata=metadata)


def pad(values, ndim, fill):
    """pad 3d values for images with more than 3 dimensions"""
    values = list(values)
    return values + [fill] * (ndim - len(values))

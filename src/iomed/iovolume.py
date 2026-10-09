import SimpleITK as sitk
import numpy as np


def write_volume(filename, array, *, spacing=None, origin=None, transform=None, metadata=None, use_compression=True):
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.bool_):
        array = array.astype('uint8')
    im = sitk.GetImageFromArray(array.T)
    if spacing is not None:
        im.SetSpacing(spacing)
    if origin is not None:
        im.SetOrigin(origin)
    if transform is not None:
        im.SetDirection(np.asarray(transform).ravel(order='F'))
    if metadata is not None:
        for key, value in metadata.items():
            im.SetMetaData(key, value)    
    sitk.WriteImage(im, filename, useCompression=use_compression)


def read_image(filename):
    im = sitk.ReadImage(filename)

    array = sitk.GetArrayFromImage(im).T
    spacing = im.GetSpacing()
    origin = im.GetOrigin()
    transform = im.GetDirection()
    metadata = {key: im.GetMetaData(key) for key in im.GetMetaDataKeys()}

    header = dict(spacing=spacing, origin=origin, transform=transform, metadata=metadata)
    return array, header
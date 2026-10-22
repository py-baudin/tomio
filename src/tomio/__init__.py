""" tomio: I/O utilities for medical images

    import tomio as io
    vol = io.read_volume("vol.nii.gz")
    cfg = io.read_config("config.yml")
    labels = io.read_labels("labels.txt")

registration (see `tomio.elastix`):
    registered, trans = io.register(fixed, moving, transform="rigid", metric="mi", return_transform=True)
    roi_registered = io.warp(trans, roi, ref=fixed, method="roi")
    roi_full = io.interpolate_slices(roi, image=moving, method="registration")
"""
from . import volume, configfile, labelfile, interp, elastix

read_volume = volume.read
write_volume = volume.write
tovolume = volume.tovolume
asvolume = volume.asvolume
resample = volume.resample
Volume = volume.Volume
Geometry = volume.Geometry

read_config = configfile.read
write_config = configfile.write

read_labels = labelfile.read
write_labels = labelfile.write

register = elastix.register
warp = elastix.transform  # not `transform`: the direction attribute of volumes
interpolate_slices = elastix.interpolate_slices
check_folding = elastix.check_folding

""" tomio: I/O utilities for medical images

    import tomio as io
    vol = io.read_volume("vol.nii.gz")
    cfg = io.read_config("config.yml")
    labels = io.read_labels("labels.txt")
"""
from . import volume, configfile, labelfile, interp

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

from . import volume, configfile, labelfile, interp

# backward compatible name
config = configfile

volread = volume.read
volwrite = volume.write
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


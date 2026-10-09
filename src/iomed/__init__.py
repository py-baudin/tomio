from . import volume, configfile, labelfile

# backward compatible name
config = configfile

volread = volume.read
volwrite = volume.write
read_volume = volume.read
write_volume = volume.write
tovolume = volume.tovolume
asvolume = volume.asvolume
resample = volume.resample

read_config = configfile.read
write_config = configfile.write

read_labels = labelfile.read
write_labels = labelfile.write


def __getattr__(name):
    # lazy import: interp needs scipy, which is optional
    if name == "interp":
        import importlib
        return importlib.import_module(".interp", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from . import volume, config, labelfile

volread = volume.read
volwrite = volume.write
read_volume = volume.read
write_volume = volume.write
tovolume = volume.tovolume
asvolume = volume.asvolume

read_config = config.read
write_config = config.write

read_labels = labelfile.read
write_labels = labelfile.write


def __getattr__(name):
    # lazy import: interp needs scipy, which is optional
    if name == "interp":
        import importlib
        return importlib.import_module(".interp", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


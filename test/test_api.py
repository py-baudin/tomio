""" top-level API: import tomio as io """
import numpy as np

import tomio as io
from tomio import volume, configfile, labelfile


def test_api_names():
    assert io.read_volume is volume.read
    assert io.write_volume is volume.write
    assert io.read_config is configfile.read
    assert io.write_config is configfile.write
    assert io.read_labels is labelfile.read
    assert io.write_labels is labelfile.write
    assert io.Volume is volume.Volume
    assert io.Geometry is volume.Geometry
    assert io.resample is volume.resample
    # removed names
    for name in ["config", "volread", "volwrite"]:
        assert not hasattr(io, name)


def test_api_usage(tmp_path):
    vol = io.Volume(np.random.rand(3, 4, 5).astype(np.float32), spacing=[1, 2, 3])
    io.write_volume(tmp_path / "vol.nii.gz", vol)
    vol2 = io.read_volume(tmp_path / "vol.nii.gz")
    assert np.array_equal(vol2, vol)
    assert vol2.spacing == (1, 2, 3)

    io.write_config(tmp_path / "config.yml", {"a": 1, "b": [1.5, "x"]})
    assert io.read_config(tmp_path / "config.yml") == {"a": 1, "b": [1.5, "x"]}

    io.write_labels(tmp_path / "labels.txt", {1: "a", 2: "b"})
    assert io.read_labels(tmp_path / "labels.txt") == {1: "a", 2: "b"}

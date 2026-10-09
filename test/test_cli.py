import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from iomed import cli, volume, interp
Volume = volume.Volume

GEOMETRY = dict(spacing=[1, 2, 3], origin=[4, 5, 6], transform=((0.8, 0.6, 0), (-0.6, 0.8, 0), (0, 0, 1)))


@pytest.fixture
def vol_file(tmp_path):
    filename = tmp_path / "vol.mha"
    volume.write(filename, Volume(np.random.rand(6, 7, 8).astype(np.float32), metadata={"foo": "bar"}, **GEOMETRY))
    # same input as the command line
    return filename, volume.read(filename)


def run(*args):
    return cli.main([str(arg) for arg in args])


# convert

@pytest.mark.parametrize("ext", [".nii.gz", ".mha", ".nrrd"])
def test_convert(tmp_path, vol_file, ext):
    filename, vol = vol_file
    output = tmp_path / f"out{ext}"
    assert run("convert", filename, output) == 0
    out = volume.read(output)
    assert np.array_equal(out, vol)
    assert out.geometry.same_as(vol)
    if ext == ".mha":
        assert out.metadata["foo"] == "bar"


def test_convert_dtype_compression(tmp_path, vol_file):
    filename, vol = vol_file
    assert run("convert", filename, tmp_path / "out.nii.gz", "--dtype", "float64", "--compression-level", 9) == 0
    out = volume.read(tmp_path / "out.nii.gz")
    assert out.dtype == np.float64
    assert np.allclose(out, vol)
    assert out.geometry.same_as(vol)


def test_convert_vector_and_complex(tmp_path):
    # vector image: stays a vector image
    im = sitk.GetImageFromArray(np.random.rand(5, 4, 3, 3).astype(np.float32), isVector=True)
    sitk.WriteImage(im, str(tmp_path / "rgb.mha"))
    assert run("convert", tmp_path / "rgb.mha", tmp_path / "rgb.nii.gz", "--dtype", "float64") == 0
    out = sitk.ReadImage(str(tmp_path / "rgb.nii.gz"))
    assert out.GetNumberOfComponentsPerPixel() == 3
    assert np.allclose(sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(im))

    # complex (written by iomed)
    cvol = Volume(np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5), **GEOMETRY)
    volume.write(tmp_path / "complex.mha", cvol)
    assert run("convert", tmp_path / "complex.mha", tmp_path / "complex.nii.gz") == 0
    assert np.allclose(volume.read(tmp_path / "complex.nii.gz", as_complex=True), cvol)


def test_convert_force(tmp_path, vol_file, capsys):
    filename, vol = vol_file
    output = tmp_path / "out.nii.gz"
    output.write_text("existing")
    assert run("convert", filename, output) == 1
    assert "--force" in capsys.readouterr().err
    assert output.read_text() == "existing"
    assert run("convert", filename, output, "--force") == 0
    assert np.array_equal(volume.read(output), vol)


# resample

def test_resample_spacing(tmp_path, vol_file):
    filename, vol = vol_file
    assert run("resample", filename, tmp_path / "out.nii.gz", "--spacing", 0.5, 1, 1.5) == 0
    out = volume.read(tmp_path / "out.nii.gz")
    expected = volume.resample(vol, spacing=(0.5, 1, 1.5))
    assert out.shape == expected.shape
    assert np.allclose(out, expected, atol=1e-6)
    assert out.geometry.same_as(expected)


def test_resample_like(tmp_path, vol_file):
    filename, vol = vol_file
    ref = Volume(np.zeros((5, 5, 5), dtype=np.uint8), spacing=[2, 2, 2], origin=[3, 4, 5])
    volume.write(tmp_path / "ref.nii.gz", ref)
    assert run("resample", filename, tmp_path / "out.mha", "--like", tmp_path / "ref.nii.gz", "--method", "linear", "--fill", "nan") == 0
    out = volume.read(tmp_path / "out.mha")
    expected = volume.resample(vol, like=ref, method="linear", fill=np.nan)
    assert out.geometry.same_as(ref)
    assert np.allclose(out, expected, atol=1e-6, equal_nan=True)
    assert np.any(np.isnan(out))


def test_resample_labels(tmp_path):
    volume.write(tmp_path / "labels.nii.gz", Volume(np.random.randint(0, 3, (6, 7, 8)).astype(np.uint8), **GEOMETRY))
    # same input as the command line: the geometry stored in NIfTI is float32 (rounding of nearest ties)
    labelmap = volume.read(tmp_path / "labels.nii.gz")
    # default: nearest for integer data
    assert run("resample", tmp_path / "labels.nii.gz", tmp_path / "out1.nii.gz", "--spacing", 0.5, 1, 1.5) == 0
    out = volume.read(tmp_path / "out1.nii.gz")
    assert out.dtype == np.uint8
    assert np.array_equal(out, volume.resample(labelmap, spacing=(0.5, 1, 1.5)))
    for method in ["labels", "roi"]:
        output = tmp_path / f"out_{method}.nii.gz"
        assert run("resample", tmp_path / "labels.nii.gz", output, "--spacing", 0.5, 1, 1.5, "--method", method) == 0
        assert np.array_equal(volume.read(output), volume.resample(labelmap, spacing=(0.5, 1, 1.5), method=method))


def test_resample_complex(tmp_path):
    volume.write(tmp_path / "complex.mha", Volume((np.random.rand(6, 7, 8) + 1j * np.random.rand(6, 7, 8)).astype(np.complex64), **GEOMETRY))
    cvol = volume.read(tmp_path / "complex.mha", as_complex=True)
    assert run("resample", tmp_path / "complex.mha", tmp_path / "out.mha", "--spacing", 2, 2, 2, "--method", "linear") == 0
    out = volume.read(tmp_path / "out.mha", as_complex=True)
    assert np.iscomplexobj(out)
    expected = volume.resample(cvol.real, spacing=(2, 2, 2), method="linear")
    assert np.allclose(out.real, expected, atol=1e-5)


def test_resample_4d_two_timepoints(tmp_path):
    # last axis of size 2, not a vector image: not complex
    vol = Volume(np.random.rand(6, 7, 8, 2).astype(np.float32))
    volume.write(tmp_path / "vol4d.nii.gz", vol)
    assert run("resample", tmp_path / "vol4d.nii.gz", tmp_path / "out.nii.gz", "--spacing", 2, 2, 2) == 1


# info

def test_info(vol_file, capsys):
    filename, vol = vol_file
    assert run("info", filename, "--metadata") == 0
    out = capsys.readouterr().out
    assert "6 x 7 x 8" in out
    assert "32-bit float" in out
    assert "(1, 2, 3)" in out
    assert "(4, 5, 6)" in out
    assert vol.get_orientation() in out
    assert vol.get_orientation(convention="to") in out
    assert "foo = bar" in out


# errors

def test_errors(tmp_path, vol_file, capsys):
    filename, _ = vol_file
    assert run("info", tmp_path / "missing.mha") == 1
    err = capsys.readouterr().err
    assert err.startswith("iomed: error:") and "missing.mha" in err and "Traceback" not in err

    assert run("convert", filename, tmp_path / "out.unknown") == 1
    assert run("convert", filename, tmp_path / "out.nii.gz", "--dtype", "foo") == 1
    assert "invalid dtype" in capsys.readouterr().err

    # usage errors (argparse): exit code 2
    assert run("resample", filename, tmp_path / "out.nii.gz") == 2  # --like or --spacing required
    assert run("resample", filename, tmp_path / "out.nii.gz", "--spacing", 1, 1) == 2
    assert run("unknown") == 2
    assert run() == 2


def test_entry_point():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-m", "iomed", "--help"], capture_output=True, text=True, env=env)
    assert result.returncode == 0
    for command in ["convert", "resample", "info"]:
        assert command in result.stdout

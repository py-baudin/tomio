from pathlib import Path
import numpy as np
import pytest
import yaml

from iomed import config


def test_tame():
    tamed = config.tame(
        {
            "flag": True,
            "np": np.True_,
            "i": np.int64(3),
            "f": np.float32(0.5),
            "z": np.complex64(1 + 2j),
            "t": (1, 2),
            "arr": np.arange(3),
            "scalar": np.array(2.0),
            "path": Path("a"),
            "none": None,
        }
    )
    assert tamed["flag"] is True
    assert tamed["np"] is True
    assert type(tamed["i"]) is int and tamed["i"] == 3
    assert type(tamed["f"]) is float and tamed["f"] == 0.5
    assert tamed["z"] == "1 + 2i"
    assert tamed["t"] == [1, 2]
    assert tamed["arr"] == [0, 1, 2]
    assert all(type(v) is int for v in tamed["arr"])
    assert type(tamed["scalar"]) is float and tamed["scalar"] == 2.0
    assert tamed["path"] == "a"
    assert tamed["none"] is None


@pytest.mark.parametrize("ext", [".json", ".yml"])
def test_config_roundtrip(tmp_path, ext):
    cfg = {
        "flag": True,
        "np": np.True_,
        "z": 1 + 2j,
        "t": (1, 2),
        "nested": {"a": [1.5, None, "s"]},
        "arr": np.arange(3),
        "arr2d": np.ones((2, 3)),
        "carr": np.array([1 + 1j, 2]),
    }
    path = tmp_path / f"config{ext}"
    config.write(path, cfg)
    out = config.read(path)

    assert out["flag"] is True
    assert out["np"] is True
    assert out["z"] == 1 + 2j
    assert out["t"] == [1, 2]
    assert out["nested"] == {"a": [1.5, None, "s"]}
    # arrays are saved as lists
    assert out["arr"] == [0, 1, 2]
    assert out["arr2d"] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert out["carr"] == [1 + 1j, 2 + 0j]
    assert type(out["carr"][1]) is complex


@pytest.mark.parametrize("ext", [".json", ".yml"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_config_complex_ndarray(tmp_path, ext, dtype):
    arr = np.array([[1 + 1j, 2], [-3.5j, 0.1 - 4j]], dtype=dtype)
    path = tmp_path / f"config{ext}"
    config.write(path, {"arr": arr, "scalar": arr[0, 0], "zero_d": np.array(2j, dtype=dtype)})
    out = config.read(path)

    assert out["arr"][1][0] == -3.5j
    assert all(type(v) is complex for row in out["arr"] for v in row)
    assert np.array_equal(np.asarray(out["arr"], dtype=dtype), arr)
    assert out["scalar"] == 1 + 1j
    assert out["zero_d"] == 2j


@pytest.mark.parametrize(
    "value, string",
    [
        (3 + 4j, "3 + 4i"),
        (1.5 - 2j, "1.5 - 2i"),
        (complex(-0.25, 0), "-0.25 + 0i"),
        (complex(0, -1e-10), "0 - 1e-10i"),
        (complex(float("inf"), 1), "inf + 1i"),
    ],
)
def test_complex_format(value, string):
    assert config.format_complex(value) == string
    assert config.parse_complex(string) == value


@pytest.mark.parametrize("string", ["hello", "3+4", "3 + 4j", "i", "1 + 2i + 3i", "a + 2i"])
def test_complex_parse_other_strings(string):
    assert config.parse_complex(string) is None
    assert config.untame({"s": string}) == {"s": string}


def test_complex_parse_formats():
    assert config.parse_complex("3+4i") == 3 + 4j
    assert config.parse_complex(" -1.5e2 - .5i ") == complex(-150, -0.5)


@pytest.mark.parametrize(
    "name, kind, expected",
    [
        ("c.json", "json", "c.json"),
        ("c.yml", "json", "c.yml"),  # suffix wins over kind
        ("c.yaml", "json", "c.yaml"),
        ("c.YAML", "json", "c.YAML"),
        ("c", "json", "c.json"),
        ("c", "yaml", "c.yaml"),
        ("c", "yml", "c.yml"),
    ],
)
def test_config_extensions(tmp_path, name, kind, expected):
    config.write(tmp_path / name, {"a": 1}, kind=kind)
    assert [p.name for p in tmp_path.iterdir()] == [expected]
    assert config.read(tmp_path / expected) == {"a": 1}


def test_config_read_no_suffix(tmp_path):
    # no suffix: json
    config.write(tmp_path / "c.json", {"a": 1})
    assert config.read(tmp_path / "c") == {"a": 1}


def test_config_unknown_type(tmp_path):
    with pytest.raises(ValueError):
        config.write(tmp_path / "c.txt", {"a": 1})
    with pytest.raises(ValueError):
        config.write(tmp_path / "c", {"a": 1}, kind="txt")
    (tmp_path / "c.txt").write_text("a: 1")
    with pytest.raises(ValueError):
        config.read(tmp_path / "c.txt")
    assert not (tmp_path / "c").exists()


def test_yaml_unsafe(tmp_path):
    path = tmp_path / "c.yml"
    path.write_text("a: !!python/object/apply:os.system ['echo unsafe']\n")
    with pytest.raises(yaml.YAMLError):
        config.read(path)


def test_yaml_globals_unchanged():
    # importing iomed.config does not register tags in PyYAML's default classes
    assert "!ndarray" not in yaml.dump(np.arange(3))

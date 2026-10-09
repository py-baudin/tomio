# -*- coding: utf-8 -*-
""" test io.Labels """

import types
import numpy as np
import pytest

from iomed import labelfile

Labels = labelfile.Labels


def test_labels_class(tmpdir):
    """test ROI label class"""

    init = {1: "label-1", 2: "label-2"}
    labels = Labels(init)
    assert len(labels) == 2
    assert list(labels) == [1, 2]
    assert labels[1] == "label-1"
    assert labels[2] == "label-2"
    assert labels.getindex("label-1") == 1
    assert labels.colors[1]
    assert labels.colors[2]

    # test saving labels
    filename = tmpdir.join("labels.txt")
    Labels.save(filename, labels)

    # test loading labels and compare
    labels2 = Labels.load(filename)
    assert labels2 == labels

    init = {4: {"LABEL": "label_4", "RGBA": (4, 3, 2, 1)}}
    labels = Labels(init)
    assert len(labels) == 1
    assert list(labels) == [4]
    assert labels[4] == "label_4"
    assert labels.colors[4] == (4, 3, 2, 1)

    with pytest.raises(ValueError):
        # non string index
        Labels({1: 2})

    with pytest.raises(ValueError):
        # invalid syntax index
        Labels({"hello:": 2})

    with pytest.raises(ValueError):
        # duplicates
        Labels({1: "a", 2: "a"})

    # test load file
    test_file_content = (
        " ## header #\n"
        '0     0    0    0        0  0  0    "Clear Label"\n'
        '4   250  255  228        1  1  1    "Some Label"\n'
        "\n"
    )
    filename = tmpdir.join("labels.txt")
    with open(filename, "w") as f:
        f.write(test_file_content)

    labels = Labels.load(filename)
    assert len(labels) == 2
    assert labels[0] == "Clear Label"
    assert labels[4] == "Some Label"
    assert labels.colors[0] == (0, 0, 0, 0)
    assert labels.colors[4] == (250, 255, 228, 1)

    with pytest.raises(IOError):
        Labels.load("unknown")

    # test save file
    labels[4] = "FOOBAR"
    filename2 = tmpdir.join("labels2.txt")
    Labels.save(filename2, labels)

    with open(filename2, "r") as f:
        lines = f.read()
        assert '"FOOBAR"' in lines


@pytest.mark.parametrize(
    "descr", ["Left hippocampus (CA1)", "WM/GM", "a,b", "label #2", "Côté", ""]
)
def test_labels_roundtrip(tmp_path, descr):
    labels = Labels({1: descr})
    filename = tmp_path / "labels.txt"
    labelfile.write(filename, labels)
    assert labelfile.read(filename) == labels


def test_labels_order():
    labels = Labels({1: "a", 2: "b"})
    labels[1] = "x"
    assert labels.indices == [1, 2]
    assert labels.descriptions == ["x", "b"]
    assert list(labels.keys()) == [1, 2]
    assert list(labels.values()) == ["x", "b"]
    assert list(labels.items()) == [(1, "x"), (2, "b")]


def test_labels_copy():
    labels = Labels({1: "a"}, allow_duplicates=True)
    copy = Labels(labels)
    copy[1] = "new"
    assert labels[1] == "a"
    assert copy.allow_duplicates


def test_labels_invalid_update_keeps_label():
    labels = Labels({1: "a", 2: "b"})
    color = labels.colors[1]
    for descr in ['a"b', "b"]:  # invalid, duplicate
        with pytest.raises(ValueError):
            labels[1] = descr
        assert labels[1] == "a"
        assert labels.colors[1] == color


def test_labels_visibility_roundtrip(tmp_path):
    labels = Labels({1: {"LABEL": "a", "VIS": 0, "MSH": 0}})
    filename = tmp_path / "labels.txt"
    labelfile.write(filename, labels)
    assert labelfile.read(filename).data[1]["VIS"] == 0
    assert labelfile.read(filename).data[1]["MSH"] == 0


LINE = '1   250  255  228        1  1  1    "a"\n'


@pytest.mark.parametrize("alpha", ["0", "1", "0.5", ".5", "1.", "1e-1"])
def test_labels_load_alpha(tmp_path, alpha):
    filename = tmp_path / "labels.txt"
    filename.write_text(f'1  250  255  228  {alpha}  1  1  "a"\n')
    assert labelfile.read(filename).colors[1][3] == float(alpha)


@pytest.mark.parametrize(
    "content, line",
    [
        ('1   250  255  228        1  |  1    "a"\n', 2),  # invalid visibility
        (LINE + LINE.replace('"a"', '"b"'), 3),  # duplicate index
        ('1   256  255  228        1  1  1    "a"\n', 2),  # invalid color
    ],
)
def test_labels_load_errors(tmp_path, content, line):
    filename = tmp_path / "labels.txt"
    filename.write_text("# header\n" + content)
    with pytest.raises(labelfile.LabelFileError, match=f"line {line}:"):
        labelfile.read(filename)


def test_labels_load_error_types():
    # backward compatible: was RuntimeError
    assert issubclass(labelfile.LabelFileError, RuntimeError)
    assert issubclass(labelfile.LabelFileError, ValueError)


@pytest.mark.parametrize("index", ["1", -1, 1.5, True, None])
def test_labels_invalid_index(index):
    with pytest.raises(ValueError):
        Labels({}).set(index, "a")


def test_labels_numpy_values():
    labels = Labels({np.int64(3): {"LABEL": "a", "RGBA": np.array([1, 2, 3, 1.0])}})
    assert labels.indices == [3]
    assert type(labels.indices[0]) is int
    assert labels.colors[3] == (1, 2, 3, 1.0)
    assert all(type(v) is int for v in labels.colors[3][:3])


@pytest.mark.parametrize(
    "info",
    [
        {"LABEL": "a", "RGBA": (256, 0, 0, 1)},
        {"LABEL": "a", "RGBA": (-1, 0, 0, 1)},
        {"LABEL": "a", "RGBA": (0.5, 0, 0, 1)},
        {"LABEL": "a", "RGBA": (True, 0, 0, 1)},
        {"LABEL": "a", "RGBA": (0, 0, 0, 1.5)},
        {"LABEL": "a", "RGBA": (0, 0, 0, -0.1)},
        {"LABEL": "a", "RGBA": (0, 0, 0)},
        {"LABEL": "a", "RGBA": "red"},
        {"LABEL": "a", "VIS": 2},
        {"LABEL": "a", "rgba": (0, 0, 0, 1)},  # unknown field
        {"RGBA": (0, 0, 0, 1)},  # missing description
    ],
)
def test_labels_invalid_info(info):
    with pytest.raises(ValueError):
        Labels({1: info})


def test_labels_palette():
    assert len(Labels({1: "a", 255: "b"}).palette) == 256 * 3
    with pytest.raises(ValueError):
        Labels({256: "a"}).palette


def test_labels_init():
    assert len(Labels()) == 0
    assert Labels(types.MappingProxyType({1: "a"}))[1] == "a"
    with pytest.raises(TypeError):
        Labels(3)
    with pytest.raises(TypeError):
        Labels("abc")
    with pytest.raises(ValueError):
        Labels([(1, "a")])  # invalid label info


def test_labels_init_sequence():
    labels = Labels(["background", "a", {"LABEL": "b", "RGBA": (1, 2, 3, 1.0)}])
    assert dict(labels) == {0: "background", 1: "a", 2: "b"}
    assert labels.colors[2] == (1, 2, 3, 1.0)
    # indices and colors arguments
    labels = Labels(["a", "b"], indices=[3, 7], colors=[(1, 2, 3, 1.0), (4, 5, 6, 0.5)])
    assert dict(labels) == {3: "a", 7: "b"}
    assert labels.colors == {3: (1, 2, 3, 1.0), 7: (4, 5, 6, 0.5)}


def test_labels_setters():
    labels = Labels({1: "a", 2: "b", 5: "c"})
    colors = labels.colors

    labels.indices = [10, 20, 30]
    assert dict(labels) == {10: "a", 20: "b", 30: "c"}
    assert list(labels.colors.values()) == list(colors.values())

    labels.descriptions = ["b", "a", "c"]  # swap
    assert labels.descriptions == ["b", "a", "c"]

    labels.colors = {20: (1, 2, 3, 1.0)}  # partial dict
    assert labels.colors[20] == (1, 2, 3, 1.0)
    assert labels.colors[10] == colors[1]
    labels.colors = [(1, 1, 1, 1.0)] * 3
    assert set(labels.colors.values()) == {(1, 1, 1, 1.0)}

    # invalid values: error, labels unchanged
    before = Labels(labels)
    invalid = [
        ("indices", [1, 1, 2]),  # duplicate
        ("indices", [1, -2, 3]),
        ("indices", [1, 2]),  # length
        ("descriptions", ["x", "x", "y"]),  # duplicate
        ("descriptions", ['x"', "y", "z"]),
        ("descriptions", "xyz"),
        ("colors", [(256, 0, 0, 1.0)] * 3),
        ("colors", {99: (0, 0, 0, 1.0)}),
    ]
    for name, value in invalid:
        with pytest.raises((ValueError, KeyError)):
            setattr(labels, name, value)
        assert labels == before


def test_labels_getindex_first():
    labels = Labels({1: "a", 2: "b", 3: "a"}, allow_duplicates=True)
    assert labels.getindex("a") == 1
    with pytest.raises(KeyError):
        labels.getindex("z")


def test_labels_mutable_mapping():
    labels = Labels()
    labels.update({1: "a", 2: "b"})
    assert dict(labels) == {1: "a", 2: "b"}
    assert labels.setdefault(1, "z") == "a"
    assert labels.get(3) is None
    del labels[2]
    assert list(labels) == [1]
    labels.clear()
    assert len(labels) == 0


def test_labels_eq():
    labels = Labels({1: "a"})
    assert labels == Labels({1: "a"})
    assert labels != Labels({1: {"LABEL": "a", "RGBA": (1, 2, 3, 1)}})  # colors differ
    assert labels == {1: "a"}  # plain mapping: descriptions only
    assert labels != {1: "b"}
    assert labels != None
    assert labels != 3


def test_labels_save(tmp_path):
    labels = Labels({1: "a", 2: "b"})
    labels.save(tmp_path / "1.txt")
    Labels.save(tmp_path / "2.txt", labels)
    Labels.save(tmp_path / "3.txt", {1: "a", 2: "b"})
    labelfile.write(tmp_path / "4.txt", labels)
    for i in range(1, 5):
        assert labelfile.read(tmp_path / f"{i}.txt") == labels
    with pytest.raises(TypeError):
        Labels.save(tmp_path / "5.txt", 3)
    assert not (tmp_path / "5.txt").exists()


def test_labels_lut():
    labels = Labels({1: {"LABEL": "a", "RGBA": (255, 0, 0, 1.0)}, 3: {"LABEL": "b", "RGBA": (0, 0, 255, 0.5)}})
    lut = labels.lut
    assert lut.shape == (4, 4) and lut.dtype == np.uint8
    assert tuple(lut[1]) == (255, 0, 0, 255)
    assert tuple(lut[3]) == (0, 0, 255, 128)
    assert tuple(lut[2]) == (0, 0, 0, 0)
    labelmap = np.array([[0, 1], [3, 3]])
    rgba = labels.lut[labelmap]
    assert rgba.shape == (2, 2, 4)
    assert tuple(rgba[1, 0]) == (0, 0, 255, 128)
    assert Labels().lut.shape == (1, 4)


def test_labels_from_volume():
    labelmap = np.array([[0, 2], [5, 5]], dtype=np.uint8)
    labels = Labels.from_volume(labelmap)
    assert labels.indices == [2, 5]
    assert labels.descriptions == ["label_2", "label_5"]
    assert labels.colors[2] != labels.colors[5]
    assert Labels.from_volume(labelmap, names=["a", "b"]).descriptions == ["a", "b"]
    labels = Labels.from_volume(labelmap, names={0: "background", 5: "b"})
    assert dict(labels) == {0: "background", 2: "label_2", 5: "b"}
    assert Labels.from_volume(labelmap > 0).indices == [1]
    with pytest.raises(ValueError):
        Labels.from_volume(labelmap, names=["a"])
    with pytest.raises(ValueError):
        Labels.from_volume(labelmap.astype(float))


def test_labels_reindex():
    labels = Labels({0: "background", 3: "a", 7: {"LABEL": "b", "RGBA": (1, 2, 3, 1.0), "VIS": 0}})
    new, mapping = labels.reindex()
    assert mapping == {0: 0, 3: 1, 7: 2}
    assert dict(new) == {0: "background", 1: "a", 2: "b"}
    assert new.data[2] == labels.data[7]
    assert dict(labels) == {0: "background", 3: "a", 7: "b"}  # unchanged

    # without background
    assert Labels({3: "a", 7: "b"}).reindex()[1] == {3: 1, 7: 2}

    # apply to label map
    labelmap = np.array([[0, 3], [7, 7]], dtype=np.uint8)
    assert np.array_equal(Labels.apply_mapping(labelmap, mapping), [[0, 1], [2, 2]])
    assert np.array_equal(labelfile.apply_mapping(labelmap, mapping), [[0, 1], [2, 2]])


def test_apply_mapping():
    labelmap = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    out = labelfile.apply_mapping(labelmap, {1: 10, 4: 300, 99: 5})
    assert np.array_equal(out, [[0, 10], [2, 300]])  # unmapped values unchanged
    assert out.dtype == np.uint16  # promoted
    assert labelfile.apply_mapping(labelmap, {1: 2}).dtype == np.uint8
    for invalid in [labelmap.astype(float), labelmap.astype(int) - 1]:
        with pytest.raises(ValueError):
            labelfile.apply_mapping(invalid, {1: 2})


def test_apply_mapping_volume():
    volume = pytest.importorskip("iomed.volume")
    labelmap = volume.Volume(np.array([[[0, 3], [7, 7]]], dtype=np.uint8), spacing=[1, 2, 3])
    out = labelfile.apply_mapping(labelmap, {3: 1, 7: 2})
    assert isinstance(out, volume.Volume)
    assert out.spacing == (1, 2, 3)


FREESURFER_LUT = """\
#$Id: FreeSurferColorLUT.txt,v 1.105 2015/07/29 18:20:54 greve Exp $

#No. Label Name:                            R   G   B   A

0   Unknown                                 0   0   0   0
2   Left-Cerebral-White-Matter              245 245 245 0
3   Left-Cerebral-Cortex                    205 62  78  0
"""

SLICER_CTBL = """\
# Color table file C:/data/labels.ctbl
# 3 values
0 background 0 0 0 0
1 tissue 128 174 128 255
2 bone with spaces 241 214 145 128
"""


def test_labels_read_freesurfer(tmp_path):
    filename = tmp_path / "FreeSurferColorLUT.txt"
    filename.write_text(FREESURFER_LUT)
    labels = labelfile.read(filename, format="freesurfer")
    assert dict(labels) == {0: "Unknown", 2: "Left-Cerebral-White-Matter", 3: "Left-Cerebral-Cortex"}
    # A: transparency
    assert labels.colors[0] == (0, 0, 0, 1.0)
    assert labels.colors[3] == (205, 62, 78, 1.0)


def test_labels_read_slicer(tmp_path):
    filename = tmp_path / "labels.ctbl"
    filename.write_text(SLICER_CTBL)
    labels = labelfile.read(filename, format="slicer")
    assert dict(labels) == {0: "background", 1: "tissue", 2: "bone with spaces"}
    # A: opacity
    assert labels.colors[0] == (0, 0, 0, 0.0)
    assert labels.colors[1] == (128, 174, 128, 1.0)
    assert labels.colors[2][3] == pytest.approx(128 / 255)


def test_labels_read_format_errors(tmp_path):
    filename = tmp_path / "labels.txt"
    filename.write_text(FREESURFER_LUT)
    with pytest.raises(ValueError):
        labelfile.read(filename, format="unknown")
    # wrong format
    with pytest.raises(labelfile.LabelFileError, match="line 5"):
        labelfile.read(filename)
    filename.write_text("1 name 256 0 0 0\n")
    with pytest.raises(labelfile.LabelFileError, match="line 1"):
        labelfile.read(filename, format="slicer")


def test_labels_pop():
    labels = Labels({1: "a", 2: "b"})
    assert labels.pop(99, None) is None
    with pytest.raises(KeyError):
        labels.pop(99)
    assert labels.pop(1) == "a"
    assert list(labels) == [2]


def test_labels_invalid_description():
    for descr in ['a"b', "a\nb", "a\rb"]:
        with pytest.raises(ValueError):
            Labels({1: descr})


def test_labels_load_comments(tmp_path):
    filename = tmp_path / "labels.txt"
    filename.write_text(
        "# comment\n"
        '  # indented comment "with quotes"\n'
        '1   250  255  228        1  1  1    "label #1"\n'
        '2   250  255  228        1  1  1    "Côté"\n',
        encoding="utf-8",
    )
    labels = labelfile.read(filename)
    assert labels.descriptions == ["label #1", "Côté"]

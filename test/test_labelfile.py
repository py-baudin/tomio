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
        Labels([(1, "a")])


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
        Labels.save(tmp_path / "5.txt", [(1, "a")])


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

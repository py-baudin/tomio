# -*- coding: utf-8 -*-
""" test io.Labels """

from iomed import labelfile
import pytest

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

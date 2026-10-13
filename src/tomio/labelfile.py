# -*- coding: utf-8 -*-
""" i/o for ITKSnap-like labels """

import re
import numbers
import functools
import itertools
from collections import abc
import numpy as np


class LabelFileError(ValueError, RuntimeError):
    """invalid label file (also a RuntimeError, for backward compatibility)"""


class _hybridmethod:
    """method callable on the class: `Cls.meth(cls, ..., labels)`,
    or on an instance: `obj.meth(...)` (the instance is passed as `labels`)"""

    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, obj, cls=None):
        if obj is None:
            return functools.partial(self.func, cls)
        return functools.partial(self.func, type(obj), labels=obj)

COLOR_MAP = [
    [228, 26, 28],
    [55, 126, 184],
    [77, 175, 74],
    [152, 78, 163],
    [255, 127, 0],
    [255, 255, 51],
    [166, 86, 40],
    [247, 129, 191],
    [153, 153, 153],
]

PREAMBLE = (
    "################################################\n"
    "# Label Description File\n"
    "# File format: \n"
    "# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL\n"
    "# Fields:\n"
    "#    IDX:   Zero-based index \n"
    "#    -R-:   Red color component (0..255)\n"
    "#    -G-:   Green color component (0..255)\n"
    "#    -B-:   Blue color component (0..255)\n"
    "#    -A-:   Label transparency (0.00 .. 1.00)\n"
    "#    VIS:   Label visibility (0 or 1)\n"
    "#    MSH:   Label mesh visibility (0 or 1)\n"
    "#  LABEL:   Label description \n"
    "################################################\n"
)


FORMATS = ("itksnap", "freesurfer", "slicer")


def read(filename, **kwargs):
    """read label file (format: "itksnap" (default), "freesurfer", "slicer"; see `Labels.load`)"""
    return Labels.load(filename, **kwargs)


def write(filename, labels, **kwargs):
    return Labels.save(filename, labels, **kwargs)


class Labels(abc.MutableMapping):
    """dict-like class containing the ROI labels: {index: description}

    Indices are non-negative integers; iteration is sorted by index.
    Each label also has a color (RGBA) and visibility flags (VIS, MSH).

    # create
    labels = Labels({1: "label_1", 2: "label_2"})  # colors set automatically
    labels = Labels({1: {"LABEL": "label_1", "RGBA": (255, 0, 0, 1.0)}})
    labels = Labels.load(filename)  # or labelfile.read(filename)

    # save
    labels.save(filename)  # or Labels.save(filename, labels), labelfile.write(filename, labels)

    # plain data (e.g. JSON, YAML, config files)
    data = labels.to_dict()  # {index: {"LABEL", "RGBA", "VIS", "MSH"}}
    labels = Labels.from_dict(data)  # indices may be strings (JSON)

    # access
    name = labels[1]  # description of index 1
    index = labels.getindex("label_1")  # index of a description
    labels.indices, labels.descriptions  # sorted by index
    rgba = labels.colors[1]  # (R, G, B, A) of index 1

    # modify (same as a dict)
    labels[1] = "another name"  # keeps color
    labels.set(3, {"LABEL": "label_3", "RGBA": (0, 255, 0, 1.0)})
    del labels[3]
    """

    # any character except double quotes
    label_re_syntax = r"(?P<name>[^\"]*)"
    re_parse_line = re.compile(
        (
            r"^\s*(?P<index>\d+)"
            r"\s+(?P<R>\d+)"
            r"\s+(?P<G>\d+)"
            r"\s+(?P<B>\d+)"
            r"\s+(?P<A>(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?)"
            r"\s+(?P<vis>[01])"
            r"\s+(?P<meshvis>[01])"
            r"\s+\"" + label_re_syntax + r"\"\s*$"
        )
    )

    # color table (FreeSurfer, Slicer): index name R G B A
    re_parse_lut_line = re.compile(
        r"^\s*(?P<index>\d+)\s+(?P<name>\S.*?)"
        r"\s+(?P<R>\d+)\s+(?P<G>\d+)\s+(?P<B>\d+)\s+(?P<A>\d+)\s*$"
    )

    @classmethod
    def load(cls, filename, *, format="itksnap", **kwargs):
        """load label file

        format:
            "itksnap": ITK-SNAP label description file
            "freesurfer": FreeSurfer color table (FreeSurferColorLUT.txt; A: transparency 0..255)
            "slicer": 3D Slicer color table (.ctbl; A: opacity 0..255)
        """
        if format not in FORMATS:
            raise ValueError(f"Unknown label file format: {format} (expected one of {FORMATS})")
        labels = cls({}, **kwargs)
        with open(filename, "r", encoding="utf-8") as fp:
            lines = fp.read().splitlines()

        for num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                # empty line or comment
                continue

            # parse line
            parse = cls._parse_itksnap_line if format == "itksnap" else cls._parse_lut_line
            parsed = parse(line, format)
            if parsed is None:
                raise LabelFileError(f"{filename}, line {num}: could not parse: {line}")

            index, info = parsed
            if index in labels:
                raise LabelFileError(f"{filename}, line {num}: duplicate index: {index}")
            try:
                labels.set(index, info)
            except ValueError as exc:
                raise LabelFileError(f"{filename}, line {num}: {exc}") from None

        return labels

    @classmethod
    def from_volume(cls, labelmap, names=None, **kwargs):
        """ labels from the values of a label map (integer array), with automatic colors

        names: descriptions, as a dict {index: name} or a sequence (one per nonzero value)
            default: "label_<index>"
        index 0 (background) is included only if it is in `names` (dict)
        """
        values = np.unique(np.asarray(labelmap))
        if not (values.dtype == bool or np.issubdtype(values.dtype, np.integer)):
            raise ValueError(f"Integer label map required, not: {values.dtype}")
        indices = [int(value) for value in values if value != 0]
        if isinstance(names, abc.Mapping):
            indices = sorted(set(indices) | {0} if 0 in names else indices)
            names = [names.get(index, f"label_{index}") for index in indices]
        elif names is None:
            names = [f"label_{index}" for index in indices]
        elif len(names) != len(indices):
            raise ValueError(f"Expected {len(indices)} names (one per nonzero value), got {len(names)}")
        return cls(dict(zip(indices, names)), **kwargs)

    @classmethod
    def from_dict(cls, mapping, **kwargs):
        """ labels from plain data {index: {"LABEL", "RGBA", "VIS", "MSH"}} (see `to_dict`)

        indices may be strings of digits (e.g. read from JSON); kwargs: see `Labels`
        """
        items = {}
        for index, info in mapping.items():
            if isinstance(index, str):
                if not index.strip().isdigit():
                    raise ValueError(f"Invalid label index: {index!r}")
                index = int(index)
            if index in items:
                raise ValueError(f"Duplicate index: {index}")
            items[index] = info
        return cls(items, **kwargs)

    def to_dict(self):
        """ plain data {index: {"LABEL", "RGBA" (list), "VIS", "MSH"}}, sorted by index

        e.g. for JSON or YAML (JSON turns the indices into strings: see `from_dict`)
        """
        return {index: dict(self.data[index], RGBA=list(self.data[index]["RGBA"])) for index in self}

    @classmethod
    def _parse_itksnap_line(cls, line, format):
        match = cls.re_parse_line.match(line)
        if not match:
            return None
        rgba = (int(match["R"]), int(match["G"]), int(match["B"]), float(match["A"]))
        info = {"LABEL": match["name"], "RGBA": rgba, "VIS": int(match["vis"]), "MSH": int(match["meshvis"])}
        return int(match["index"]), info

    @classmethod
    def _parse_lut_line(cls, line, format):
        match = cls.re_parse_lut_line.match(line)
        if not match:
            return None
        alpha = int(match["A"]) / 255
        if format == "freesurfer":
            # FreeSurfer: transparency
            alpha = 1 - alpha
        rgba = (int(match["R"]), int(match["G"]), int(match["B"]), alpha)
        return int(match["index"]), {"LABEL": match["name"], "RGBA": rgba}

    @_hybridmethod
    def save(cls, filename, labels, **kwargs):
        """save labels to file

        Labels.save(filename, labels) or labels.save(filename)
        """
        if not isinstance(labels, cls):
            labels = cls(labels, **kwargs)

        with open(filename, "w", encoding="utf-8") as f:
            # write preamble
            f.write(PREAMBLE)
            for index in labels:
                info = labels.data[index]
                rgba = info["RGBA"]
                descr = info["LABEL"]
                vis = info["VIS"]
                msh = info["MSH"]
                alpha = float(rgba[3])
                line = ('{:>5} {:>5}{:>5}{:>5} {:>9}{:>3}{:>3}    "{:}" \n').format(
                    index, rgba[0], rgba[1], rgba[2], alpha, vis, msh, descr
                )
                f.write(line)

    def __init__(self, labels=None, *, indices=None, colors=None, allow_duplicates=None):
        """init label object

        labels:
            mapping {index: description or info dict}
            sequence of descriptions (or info dicts): indices 0, 1, 2...
            Labels object (copied)
        indices: new indices, in the order of the labels (see `indices` setter)
        colors: new colors (see `colors` setter)
        allow_duplicates: default False (or copied from `labels` if it is a Labels object)
        """
        if allow_duplicates is None:
            allow_duplicates = getattr(labels, "allow_duplicates", False)
        self.allow_duplicates = allow_duplicates
        self.color_gen = itertools.cycle(COLOR_MAP)
        self.data = {}

        if isinstance(labels, Labels):
            self.data = {index: dict(info) for index, info in labels.data.items()}
        elif labels is None:
            pass
        elif isinstance(labels, abc.Mapping):
            for index in sorted(labels):
                self.set(index, labels[index])
        elif isinstance(labels, abc.Sequence) and not isinstance(labels, str):
            for index, value in enumerate(labels):
                self.set(index, value)
        else:
            raise TypeError("A mapping {index: description} or a sequence of descriptions is required, not: %r" % (labels,))

        if indices is not None:
            self.indices = indices
        if colors is not None:
            self.colors = colors

    # pickle, copy: without the color generator (itertools objects are not picklable
    # from Python 3.14); a copy restarts the color cycle, as Labels(labels)
    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("color_gen", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.color_gen = itertools.cycle(COLOR_MAP)

    def _replace(self, items):
        """ replace all labels with validated (index, info) items """
        new = type(self)(allow_duplicates=self.allow_duplicates)
        for index, info in items:
            if index in new:
                raise ValueError(f"Duplicate index: {index}")
            new.set(index, info)
        self.data = new.data

    @property
    def indices(self):
        return list(self)

    @indices.setter
    def indices(self, indices):
        """ renumber the labels: new indices, in the order of the current indices """
        self._replace(zip(indices, (self.data[old] for old in self), strict=True))

    @property
    def descriptions(self):
        return list(self.data[index]['LABEL'] for index in self)

    @descriptions.setter
    def descriptions(self, descriptions):
        """ new descriptions, in the order of the indices """
        if isinstance(descriptions, str):
            raise ValueError("A sequence of descriptions is required")
        items = zip(self, descriptions, strict=True)
        self._replace((index, dict(self.data[index], LABEL=descr)) for index, descr in items)

    @property
    def colors(self):
        """return dict of colors"""
        return dict((idx, self.data[idx]["RGBA"]) for idx in self)

    @colors.setter
    def colors(self, colors):
        """ new colors: dict {index: RGBA}, or sequence of RGBA in the order of the indices """
        if isinstance(colors, abc.Mapping):
            unknown = set(colors) - set(self.data)
            if unknown:
                raise KeyError(f"Unknown label indices: {sorted(unknown)}")
            colors = [colors.get(index, self.data[index]["RGBA"]) for index in self]
        items = zip(self, colors, strict=True)
        self._replace((index, dict(self.data[index], RGBA=rgba)) for index, rgba in items)

    @property
    def palette(self):
        """return RBG palette (eg. for using with PIL)"""
        if self.data and max(self.data) > 255:
            raise ValueError("Cannot make a 256-color palette with label indices > 255")
        palette = []
        for i in range(256):
            if i in self:
                palette.extend(self.data[i]["RGBA"][:-1])
            else:
                palette.extend((0, 0, 0))
        return palette

    @property
    def lut(self):
        """ RGBA lookup table: uint8 array of shape (max index + 1, 4)

        rgba = labels.lut[labelmap]  # (..., 4) RGBA image; missing indices: (0, 0, 0, 0)
        """
        lut = np.zeros((max(self.data, default=0) + 1, 4), dtype=np.uint8)
        for index, info in self.data.items():
            red, green, blue, alpha = info["RGBA"]
            lut[index] = (red, green, blue, round(alpha * 255))
        return lut

    def __repr__(self):
        return str(dict(zip(self.indices, self.descriptions)))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # all accessors (keys, values, items, indices...) use this order
        return iter(sorted(self.data))

    def __eq__(self, other):
        if isinstance(other, Labels):
            # compare everything (descriptions, colors, visibility)
            return self.data == other.data
        elif isinstance(other, abc.Mapping):
            # compare descriptions only
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def __contains__(self, index):
        return index in self.data

    def __getitem__(self, index):
        """return label description for given index"""
        return self.data[index]["LABEL"]

    def __setitem__(self, index, descr):
        """set label description"""
        self.set(index, descr)

    def __delitem__(self, index):
        del self.data[index]

    def set(self, index, value):
        """ add or update label

        index: non-negative integer
        value: description (str), or dict with keys:
            LABEL (description, required), RGBA, VIS, MSH
        """
        if not is_integer(index) or index < 0:
            raise ValueError("Invalid label index: %r" % (index,))
        index = int(index)

        if isinstance(value, str):
            # label description only: keep existing info (copy, validated below)
            info = dict(self.data.get(index, {}), LABEL=value)
        elif isinstance(value, abc.Mapping):
            info = value
        else:
            raise ValueError("Invalid label info: %r" % (value,))

        unknown = info.keys() - {"LABEL", "RGBA", "VIS", "MSH"}
        if unknown:
            raise ValueError("Unknown label fields: %s" % sorted(unknown))
        elif "LABEL" not in info:
            raise ValueError("Missing label description (LABEL): %s" % info)

        # full description
        descr = info["LABEL"]
        vis = info.get("VIS", 1)
        meshvis = info.get("MSH", 1)

        if not isinstance(descr, str) or any(c in descr for c in '"\n\r'):
            raise ValueError("Invalid label description: %r" % descr)
        elif not self.allow_duplicates and any(
            other == descr for i, other in self.items() if i != index
        ):
            raise ValueError("Duplicate description: %s" % descr)

        rgba = info.get("RGBA")
        if rgba is None:
            rgba = (0, 0, 0, 0.0) if index == 0 else self.newcolor()

        rgba = check_rgba(rgba)

        if not all(isinstance(v, numbers.Integral) and v in (0, 1) for v in (vis, meshvis)):
            raise ValueError("Invalid visibility values: %s" % str(info))
        self.data[index] = {
            "LABEL": descr,
            "RGBA": rgba,
            "VIS": int(vis),
            "MSH": int(meshvis),
        }

    def reindex(self):
        """ renumber labels with consecutive indices 1..N (0 is kept), in index order

        returns: new Labels, mapping {old index: new index} (see `apply_mapping`)
        """
        new_indices = itertools.count(1)
        mapping = {index: 0 if index == 0 else next(new_indices) for index in self}
        labels = type(self)({}, allow_duplicates=self.allow_duplicates)
        labels.data = {mapping[index]: dict(info) for index, info in self.data.items()}
        return labels, mapping

    @staticmethod
    def apply_mapping(labelmap, mapping):
        """ replace the values of `labelmap` with `mapping` {old: new} (see `labelfile.apply_mapping`) """
        return apply_mapping(labelmap, mapping)

    def getindex(self, name):
        """return first index of given name (reverse of .__getitem__)"""
        for index in self:
            if self.data[index]["LABEL"] == name:
                return index
        raise KeyError(name)

    def newcolor(self, alpha=1.0):
        """return a different color every time"""
        return next(self.color_gen) + [alpha]


def apply_mapping(labelmap, mapping):
    """ replace the values of an integer label map with `mapping` {old: new}

    values not in `mapping` are unchanged; the dtype is promoted if needed;
    the type and header of the label map are kept (e.g. Volume)
    """
    array = np.asarray(labelmap)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Integer label map required, not: {array.dtype}")
    if array.size and array.min() < 0:
        raise ValueError("Label map with negative values")
    if any(new < 0 for new in mapping.values()):
        raise ValueError("Mapping with negative values")
    dtype = np.promote_types(array.dtype, np.min_scalar_type(max(mapping.values(), default=0)))
    maxvalue = int(array.max()) if array.size else 0
    lut = np.arange(maxvalue + 1, dtype=dtype)
    for old, new in mapping.items():
        if 0 <= old <= maxvalue:
            lut[old] = new
    result = lut[array]
    header = getattr(labelmap, "header", None)
    return type(labelmap)(result, **header) if header is not None else result


def is_integer(value):
    """integer, including numpy integers, excluding bool"""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def is_real(value):
    """real number, including numpy floats, excluding bool"""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def check_rgba(rgba):
    """validate RGBA color: RGB integers in 0..255, alpha in 0..1"""
    try:
        *rgb, alpha = rgba
    except (TypeError, ValueError):
        rgb, alpha = [], None
    valid = (
        len(rgb) == 3
        and all(is_real(v) and float(v).is_integer() and 0 <= v <= 255 for v in rgb)
        and is_real(alpha)
        and 0 <= alpha <= 1
    )
    if not valid:
        raise ValueError("Invalid RGBA color: %s" % (rgba,))
    return (*(int(v) for v in rgb), float(alpha))

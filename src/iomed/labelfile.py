# -*- coding: utf-8 -*-
""" i/o for ITKSnap-like labels """

import re
import numbers
import functools
import itertools
from collections import abc


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


def read(filename, **kwargs):
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

    @classmethod
    def load(cls, filename, **kwargs):
        """load label file"""
        labels = cls({}, **kwargs)
        with open(filename, "r", encoding="utf-8") as fp:
            lines = fp.read().splitlines()

        for num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                # empty line or comment
                continue

            # parse line
            match = cls.re_parse_line.match(line)
            if not match:
                raise LabelFileError(f"{filename}, line {num}: could not parse: {line}")

            # index is the first number
            index = int(match.group("index"))
            if index in labels:
                raise LabelFileError(f"{filename}, line {num}: duplicate index: {index}")

            # color, including alpha
            rgba = (
                int(match.group("R")),
                int(match.group("G")),
                int(match.group("B")),
                float(match.group("A")),
            )
            info = {
                "LABEL": match.group("name"),
                "RGBA": rgba,
                "VIS": int(match.group("vis")),
                "MSH": int(match.group("meshvis")),
            }
            try:
                labels.set(index, info)
            except ValueError as exc:
                raise LabelFileError(f"{filename}, line {num}: {exc}") from None

        return labels

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

    def __init__(self, labels=None, *, allow_duplicates=None):
        """init label object

        labels: mapping {index: description or info dict}, or Labels object (copied)
        allow_duplicates: default False (or copied from `labels` if it is a Labels object)
        """
        if allow_duplicates is None:
            allow_duplicates = getattr(labels, "allow_duplicates", False)
        self.allow_duplicates = allow_duplicates
        self.color_gen = itertools.cycle(COLOR_MAP)

        if isinstance(labels, Labels):
            self.data = {index: dict(info) for index, info in labels.data.items()}
            return
        elif labels is None:
            labels = {}
        elif not isinstance(labels, abc.Mapping):
            raise TypeError("A mapping {index: description} is required, not: %r" % (labels,))
        self.data = {}

        for index in sorted(labels):
            self.set(index, labels[index])

    @property
    def indices(self):
        return list(self)
    
    @property
    def descriptions(self):
        return list(self.data[index]['LABEL'] for index in self)
    
    @property
    def colors(self):
        """return dict of colors"""
        return dict((idx, self.data[idx]["RGBA"]) for idx in self)

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

    def __repr__(self):
        return str(self.data)

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
            raise ValueError("Invalid label info: %s" % value)

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
        
    def getindex(self, name):
        """return index of given name (reverse of .__getitem__)"""
        return {self.data[index]["LABEL"]: index for index in self.indices}[name]

    def newcolor(self, alpha=1.0):
        """return a different color every time"""
        return next(self.color_gen) + [alpha]


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

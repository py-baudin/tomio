# coding=utf-8
""" config file i/o (JSON, YAML; TOML: read only) """
import re
import json
import math
import numbers
import yaml
from collections import abc
from pathlib import Path
import numpy as np


KINDS = {"json": "json", "yaml": "yaml", "yml": "yaml", "toml": "toml"}


def get_kind(path):
    """config file type ("json", "yaml" or "toml") from path suffix"""
    ext = path.suffix[1:].lower()
    if ext not in KINDS:
        raise ValueError(f"Unknown config file type: {path}")
    return KINDS[ext]


def read(path):
    """read config file (.json, .yaml, .yml or .toml; no suffix: .json)"""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".json")
    kind = get_kind(path)

    if kind == "toml":
        tomllib = import_tomllib()
        with open(path, "rb") as fp:
            return untame(tomllib.load(fp))

    with open(path, "r", encoding="utf-8") as fp:
        if kind == "json":
            return untame(json.load(fp))
        else:
            # safe: no Python-specific tags
            return untame(yaml.safe_load(fp))


def import_tomllib():
    """TOML parser: tomllib (Python >= 3.11), or tomli (same API)"""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            raise ImportError("Reading TOML files requires Python >= 3.11, or `tomli` (pip install tomli)") from None
    return tomllib


def write(path, config, kind="json"):
    """write config file

    The file type is set by the path suffix (.json, .yaml or .yml; .toml is read-only).
    If the path has no suffix, `kind` is used and added as suffix.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(f".{kind}")
    kind = get_kind(path)
    if kind == "toml":
        raise ValueError(f"Writing TOML files is not supported (read-only): {path}")
    config = tame(config)

    with open(path, "w", encoding="utf-8") as fp:
        if kind == "json":
            json.dump(config, fp)
        else:
            yaml.safe_dump(config, fp, default_flow_style=False)


def tame(element):
    """convert config elements to standard types

    dict, list, str, None, bool, int, float
    numpy arrays become lists, complex numbers become strings: "3 + 4i"
    """
    if element is None or isinstance(element, str):
        return element
    elif isinstance(element, (bool, np.bool_)):  # before int: bool is an int
        return bool(element)
    elif isinstance(element, numbers.Integral):
        return int(element)
    elif isinstance(element, numbers.Real):
        return float(element)
    elif isinstance(element, numbers.Complex):
        return format_complex(element)
    elif isinstance(element, np.ndarray):
        # tolist: nested lists, or a scalar for 0-d arrays
        return tame(element.tolist())
    elif isinstance(element, abc.Mapping):
        return {tame(key): tame(value) for key, value in element.items()}
    elif isinstance(element, abc.Sequence):
        return [tame(item) for item in element]
    else:
        return str(element)


def untame(element):
    """reverse of tame: convert complex number strings ("3 + 4i") to complex"""
    if isinstance(element, str):
        return parse_complex(element, default=element)
    elif isinstance(element, dict):
        return {key: untame(value) for key, value in element.items()}
    elif isinstance(element, list):
        return [untame(item) for item in element]
    return element


# complex numbers

_NUMBER = r"(?:(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|nan)"
_RE_COMPLEX = re.compile(rf"^\s*([-+]?{_NUMBER})\s*([-+])\s*({_NUMBER})i\s*$")


def format_complex(value):
    """format complex number as "3 + 4i" """
    value = complex(value)

    def fmt(x):
        return str(int(x)) if x.is_integer() else repr(x)

    sign = "-" if math.copysign(1, value.imag) < 0 else "+"
    return f"{fmt(value.real)} {sign} {fmt(abs(value.imag))}i"


def parse_complex(string, default=None):
    """parse complex number formatted as "3 + 4i" (return `default` if not one)"""
    match = _RE_COMPLEX.match(string)
    if not match:
        return default
    real, sign, imag = match.groups()
    return complex(float(real), float(sign + imag))

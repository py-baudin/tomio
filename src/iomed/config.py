# coding=utf-8
""" module defining config file i/o """
import re
import json
import math
import numbers
import yaml
from collections import abc
from pathlib import Path
import numpy as np


def read(path):
    """read config file"""
    path = Path(path)
    if not path.suffix or path.suffix == ".json":
        # json
        path = path.with_suffix(".json")
        with open(path, "r") as fp:
            return untame(json.load(fp))

    elif path.suffix == ".yml":
        # yaml (safe: no Python-specific tags)
        with open(path, "r") as fp:
            return untame(yaml.safe_load(fp))
    else:
        raise ValueError(f"Unknown config file type: {path}")


def write(path, config, kind="json"):
    """write config file"""
    path = Path(path)
    config = tame(config)
    if path.suffix:
        kind = path.suffix[1:]

    if kind == "json":
        path = path.with_suffix(".json")
        with open(path, "w") as fp:
            json.dump(config, fp)

    elif kind in ["yaml", "yml"]:
        path = path.with_suffix(".yml")
        with open(path, "w") as fp:
            yaml.safe_dump(config, fp, default_flow_style=False)

    else:
        raise ValueError(f"Unknown config file type: {kind}")


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

"""registration settings: component registry, sections, stages"""

import os
import copy
import inspect

import yaml

from .. import configfile as _configfile
from .interpolators import InterpolatorBSpline, InterpolatorLinear, InterpolatorNearestNeighbor
from .metrics import MetricMIND, MetricMeanSquares, MetricMutualInformation, MetricNormalizedGradientFields
from .optimizers import OptimizerASGD, OptimizerAuto, OptimizerLBFGS, OptimizerSGD
from .pyramid import PyramidDummy, PyramidGaussian
from .samplers import SamplerFull, SamplerGrid, SamplerRandom, SamplerRandomCoordinates
from .transforms import TransformAffine, TransformBSpline, TransformRigid, TransformTranslation


def make_stages(config=None, **sections):
    """list of Config objects (one per registration stage)"""
    if isinstance(config, (str, os.PathLike)):
        config = _configfile.read(config)
    if isinstance(config, (list, tuple)):
        if sections:
            raise ValueError("Keyword settings cannot be combined with several stages")
        if not config:
            raise ValueError("No registration stage")
        return [Config(cfg) for cfg in config]
    return [Config(config, **sections)]


# component registry: section -> {type name: class}
COMPONENTS = {
    "transform": {
        "translation": TransformTranslation,
        "rigid": TransformRigid,
        "affine": TransformAffine,
        "bspline": TransformBSpline,
    },
    "pyramid": {
        "gaussian": PyramidGaussian,
        "none": PyramidDummy,
    },
    "sampler": {
        "full": SamplerFull,
        "random": SamplerRandom,
        "random_coordinates": SamplerRandomCoordinates,
        "grid": SamplerGrid,
    },
    "interpolator": {
        "nearest": InterpolatorNearestNeighbor,
        "linear": InterpolatorLinear,
        "bspline": InterpolatorBSpline,
    },
    "metric": {
        "mse": MetricMeanSquares,
        "mi": MetricMutualInformation,
        "ngf": MetricNormalizedGradientFields,
        "mind": MetricMIND,
    },
    "optimizer": {
        "auto": OptimizerAuto,
        "lbfgs": OptimizerLBFGS,
        "sgd": OptimizerSGD,
        "asgd": OptimizerASGD,
    },
}


def get_component(section, name):
    """component class from its type name"""
    if section not in COMPONENTS:
        raise ValueError(f"Unknown section: '{section}' (expected one of: {', '.join(COMPONENTS)})")
    try:
        return COMPONENTS[section][name]
    except (KeyError, TypeError):
        choices = ", ".join(COMPONENTS[section])
        raise ValueError(f"Unknown {section} type: '{name}' (expected one of: {choices})") from None


def component_name(section, obj):
    """type name of a component (class or object)"""
    cls = obj if isinstance(obj, type) else type(obj)
    for name, comptype in COMPONENTS[section].items():
        if comptype is cls:
            return name
    raise ValueError(f"Unregistered {section}: {cls.__name__}")


def component_options(cls):
    """initialization options and their defaults, and whether other options are accepted (**kwargs)"""
    if cls.__init__ is object.__init__:
        return {}, False
    parameters = list(inspect.signature(cls.__init__).parameters.values())[1:]
    options = {par.name: par.default for par in parameters if par.kind in (par.POSITIONAL_OR_KEYWORD, par.KEYWORD_ONLY)}
    return options, any(par.kind == par.VAR_KEYWORD for par in parameters)


class Config:
    """Registration settings, one section per component

    sections:
        transform, pyramid, sampler, interpolator, metric, optimizer: component settings, given as
            - a type name, e.g. "rigid" (default options)
            - a dict with a "type" key and options, e.g. {"type": "mi", "nbin": 32}
            - a dict of options only (the current type is kept), e.g. {"maxsample": 5000}
            - a component object, e.g. TransformRigid(axis=2)
        initialize: {"method": ...}, initial translation of the first stage:
            None, "geometry" (align the image centers) or "moments" (align the centers of mass)
        resample: {"method": ...}, interpolation of the registered image (see `transform`)
        mask: {"fixed": ...}, mask of the fixed image when none is given: None (whole image), "auto" (foreground
            for non-global transforms, e.g. B-splines; none for global transforms) or "foreground" (always);
            the foreground is an Otsu threshold of the image (see `foreground_mask`)

    examples:
        Config(transform="rigid", metric="mi")
        Config({"transform": {"type": "rigid", "axis": 2}, "metric": "mi"})
        Config("config.yml")
        Config.describe()  # available types and options
    """

    SECTIONS = tuple(COMPONENTS) + ("initialize", "resample", "mask")
    # sections with a single option (a value or a dict): option name, allowed values (None: any)
    SIMPLE = {
        "initialize": ("method", (None, "geometry", "moments")),
        "resample": ("method", None),
        "mask": ("fixed", (None, "auto", "foreground")),
    }
    INITIALIZE = SIMPLE["initialize"][1]

    DEFAULTS = {
        "transform": "rigid",
        "pyramid": "gaussian",
        "sampler": "random_coordinates",
        "interpolator": "linear",
        "metric": "mse",
        "optimizer": "auto",
        "initialize": {"method": None},
        "resample": {"method": None},
        "mask": {"fixed": None},
    }

    def __init__(self, config=None, **sections):
        if isinstance(config, Config):
            config = config.to_dict()
        elif isinstance(config, (str, os.PathLike)):
            config = _configfile.read(config)
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"Invalid configuration: {config!r}")
        self._sections = {}
        for name, value in self.DEFAULTS.items():
            self[name] = value
        for name, value in {**(config or {}), **sections}.items():
            self[name] = value

    def __getitem__(self, name):
        value = self._sections[name]
        return value if isinstance(value, dict) is False else dict(value)

    def __setitem__(self, name, value):
        if name not in self.SECTIONS:
            raise ValueError(f"Unknown section: '{name}' (expected one of: {', '.join(self.SECTIONS)})")
        if name in self.SIMPLE:
            option, allowed = self.SIMPLE[name]
            value = {option: value} if value is None or isinstance(value, str) else dict(value)
            unknown = set(value) - {option}
            if unknown:
                raise ValueError(f"Unknown {name} option(s): {', '.join(sorted(unknown))}")
            if allowed is not None and value.get(option) not in allowed:
                raise ValueError(f"Unknown {name} {option}: {value[option]} (expected one of: {allowed})")
            self._sections[name] = {**self._sections.get(name, {}), **value}
            return
        if isinstance(value, str):
            value = {"type": value}
        elif not isinstance(value, dict):
            # component object
            component_name(name, value)  # registered type
            self._sections[name] = value
            return
        value = dict(value)
        if "type" not in value:
            # options of the current type
            current = self._sections.get(name)
            if not isinstance(current, dict):
                raise ValueError(f"No type for section '{name}'")
            value = {**current, **value}
        cls = get_component(name, value["type"])
        options, any_option = component_options(cls)
        if not any_option:
            unknown = set(value) - {"type"} - set(options)
            if unknown:
                raise ValueError(
                    f"Unknown option(s) for {name} '{value['type']}': {', '.join(sorted(unknown))}"
                    f" (expected: {', '.join(options) or 'none'})"
                )
        self._sections[name] = value

    def __contains__(self, name):
        return name in self._sections

    def __eq__(self, other):
        return isinstance(other, Config) and self.to_dict() == other.to_dict()

    def to_dict(self):
        """settings as a dict (component objects are kept as is)"""
        return {name: (dict(value) if isinstance(value, dict) else value) for name, value in self._sections.items()}

    def __repr__(self):
        return f"Config({self.to_dict()!r})"

    def __str__(self):
        data = {
            name: (value if isinstance(value, dict) else {"type": component_name(name, value), "object": repr(value)})
            for name, value in self.to_dict().items()
        }
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=None).strip()

    def setup(self, name):
        """instantiate the component of a section"""
        if name not in COMPONENTS:
            raise ValueError(f"Unknown component: '{name}' (expected one of: {', '.join(COMPONENTS)})")
        value = self._sections[name]
        if not isinstance(value, dict):
            # component object: fresh copy (components may hold state, e.g. samplers)
            return copy.deepcopy(value)
        options = dict(value)
        cls = get_component(name, options.pop("type"))
        return cls(**options)

    @classmethod
    def describe(cls):
        """available components and their options (text)"""
        lines = []
        for section, types in COMPONENTS.items():
            lines.append(f"{section}: (default: {cls.DEFAULTS[section]})")
            for name, comptype in types.items():
                doc = (comptype.__doc__ or "").strip().splitlines()
                lines.append(f"    {name}: {doc[0] if doc else comptype.__name__}")
                options, any_option = component_options(comptype)
                for option, default in options.items():
                    default = "" if default is inspect.Parameter.empty else f" = {default!r}"
                    lines.append(f"        {option}{default}")
                if any_option:
                    lines.append("        (other options passed through)")
        lines.append("initialize:")
        lines.append("    method = None  # None, 'geometry' (image centers) or 'moments' (centers of mass)")
        lines.append("resample:")
        lines.append("    method = None  # see `transform`")
        lines.append("mask:")
        lines.append("    fixed = None  # None, 'auto' (foreground for non-global transforms) or 'foreground'")
        return "\n".join(lines)

""" command-line interface

    tomio convert IN OUT
    tomio resample IN OUT (--like REF | --spacing SX SY SZ) [--method METHOD]
    tomio info IN
    tomio register FIXED MOVING [MOVING ...] [-c CONFIG] [-o TEMPLATE] [--save-transform] [--plot]
    tomio warp TRANSFORM MOVING [MOVING ...] [--ref REF] [-o TEMPLATE] [--method METHOD]
    tomio interpolate-slices ROI OUT [--image IMAGE] [--method shape|registration]
    tomio register-config [FILE]

output name templates: <moving>, <fixed> (file names without extension), <date>, <datetime>
"""
import sys
import argparse
import datetime
from pathlib import Path

import numpy as np

from . import volume, imagefile, configfile

METHODS = ["nearest", "linear", "spline3", "labels", "roi"]


class UserError(Exception):
    """error reported to the user without traceback"""


def main(argv=None):
    """run the command line, return the exit code"""
    parser = get_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # --help, or invalid arguments (argparse message already printed)
        return exc.code
    try:
        args.func(args)
    except (UserError, FileNotFoundError, ValueError, RuntimeError, ImportError, NotImplementedError) as exc:
        print(f"tomio: error: {exc}", file=sys.stderr)
        return 1
    return 0


def get_parser():
    parser = argparse.ArgumentParser(prog="tomio", description="Convert, resample, inspect and register medical image volumes.")
    commands = parser.add_subparsers(title="commands", required=True, metavar="COMMAND")

    # convert
    convert = commands.add_parser("convert", help="convert an image file to another format", description=(
        "Convert an image file to another format (from the file extensions, e.g. .mha, .nii.gz, .nrrd). "
        "Pixel data, geometry, metadata, vector and complex pixels are kept."))
    convert.add_argument("input", help="input image file")
    convert.add_argument("output", help="output image file")
    add_output_options(convert)
    convert.set_defaults(func=run_convert)

    # resample
    resample = commands.add_parser("resample", help="resample a volume to a new geometry", description=(
        "Resample a volume to the geometry of another volume, or to a new spacing (same extent)."))
    resample.add_argument("input", help="input image file")
    resample.add_argument("output", help="output image file")
    target = resample.add_mutually_exclusive_group(required=True)
    target.add_argument("--like", metavar="REF", help="reference image file: use its geometry (header only)")
    target.add_argument("--spacing", metavar=("SX", "SY", "SZ"), nargs=3, type=float, help="new spacing")
    resample.add_argument("--method", choices=METHODS, help=(
        "interpolation method; default: nearest for integer data, spline3 otherwise; "
        "labels: label maps, label by label; roi: masks/ROIs"))
    resample.add_argument("--fill", type=float, default=0.0, help="value outside the input volume (default: 0, may be nan)")
    add_output_options(resample)
    resample.set_defaults(func=run_resample)

    # info
    info = commands.add_parser("info", help="print the header of an image file", description=(
        "Print shape, pixel type and geometry of an image file (the pixel data is not read)."))
    info.add_argument("input", help="input image file")
    info.add_argument("--metadata", action="store_true", help="also print the metadata")
    info.set_defaults(func=run_info)

    # register
    register = commands.add_parser("register", help="register moving volume(s) onto a fixed volume", description=(
        "Register moving volume(s) onto a fixed volume (tomio.elastix): rigid, affine, B-spline (free-form) "
        "transforms, MI/MSE/NGF/MIND metrics, successive stages. Settings: a configuration file (-c, see "
        "'tomio register-config') and/or --transform/--metric. Output name templates: <moving>, <fixed>, "
        "<date>, <datetime>."))
    register.add_argument("fixed", help="fixed image file")
    register.add_argument("moving", nargs="+", help="moving image file(s)")
    register.add_argument("-c", "--config", help="registration settings (YAML/JSON file; a list: successive stages)")
    register.add_argument("--transform", help="transform type (single stage): translation, rigid, affine, bspline")
    register.add_argument("--metric", help="metric (single stage): mse, mi, ngf, mind")
    register.add_argument("-o", "--output", default="reg_<moving>", help="output file template (default: reg_<moving>, extension of the moving file)")
    register.add_argument("-t", "--save-transform", nargs="?", const="transform_<moving>.yml", metavar="TEMPLATE",
                          help="save the transformation(s) (default template: transform_<moving>.yml)")
    register.add_argument("--plot", nargs="?", const="reg_<moving>.png", metavar="TEMPLATE",
                          help="save a figure of the result (matplotlib; default template: reg_<moving>.png)")
    register.add_argument("--fixed-mask", help=(
        "mask of the fixed image (samples drawn inside): a file, 'foreground' (automatic foreground mask, all "
        "stages) or 'auto' (foreground for non-global stages)"))
    register.add_argument("--moving-mask", help="mask of the moving image(s) (samples mapped outside are discarded)")
    register.add_argument("--reset", action="store_true", help="several moving images: start each one from the initial parameters")
    register.add_argument("-f", "--force", action="store_true", help="overwrite the output files if they exist")
    register.add_argument("-v", "--verbose", action="store_true", help="print the optimization progress")
    register.set_defaults(func=run_register)

    # warp
    warp = commands.add_parser("warp", help="apply a saved transformation to volume(s)", description=(
        "Resample moving volume(s) through a transformation saved by 'tomio register --save-transform'."))
    warp.add_argument("transform", help="transformation file (YAML/JSON)")
    warp.add_argument("moving", nargs="+", help="moving image file(s)")
    warp.add_argument("--ref", help="reference image file: output grid (default: the fixed image grid of the transformation)")
    warp.add_argument("-o", "--output", default="warped_<moving>", help="output file template (default: warped_<moving>)")
    warp.add_argument("--method", choices=METHODS, help=(
        "interpolation method; default: nearest for integer data, spline3 otherwise; "
        "labels: label maps, label by label; roi: ROIs drawn on some slices"))
    warp.add_argument("-f", "--force", action="store_true", help="overwrite the output files if they exist")
    warp.set_defaults(func=run_warp)

    # interpolate-slices
    slices = commands.add_parser("interpolate-slices", help="fill a ROI drawn on some slices", description=(
        "Fill the slices between the labelled slices of a sparse ROI (label map): shape-based interpolation of "
        "each label (signed distances), optionally guided by the registration of consecutive image slices."))
    slices.add_argument("input", help="ROI file (label map)")
    slices.add_argument("output", help="output file")
    slices.add_argument("--image", help="image on the ROI grid (method 'registration')")
    slices.add_argument("--method", choices=["shape", "registration"], help="default: registration with --image, shape otherwise")
    slices.add_argument("--axis", type=int, help="slice axis (default: guessed from the ROI)")
    slices.add_argument("--drop-label", type=int, nargs="+", default=[], metavar="LABEL", help="labels set to 0 first (e.g. 'not annotated')")
    slices.add_argument("-f", "--force", action="store_true", help="overwrite the output file if it exists")
    slices.set_defaults(func=run_interpolate_slices)

    # register-config
    config = commands.add_parser("register-config", help="describe the registration settings", description=(
        "Print the registration components and their options, or write a template configuration file "
        "(rigid, then B-spline) to edit."))
    config.add_argument("output", nargs="?", help="template configuration file to write (.yml or .json)")
    config.add_argument("-f", "--force", action="store_true", help="overwrite the output file if it exists")
    config.set_defaults(func=run_register_config)
    return parser


def add_output_options(parser):
    parser.add_argument("--dtype", help="cast the output pixel values (numpy dtype, e.g. float32, uint8)")
    parser.add_argument("--compression-level", type=int, metavar="N", help="compression level (e.g. 1-9 for .nii.gz)")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite the output file if it exists")


# commands

def run_convert(args):
    check_files(args)
    imagefile.convert_image(args.input, args.output, dtype=get_dtype(args.dtype), compression_level=args.compression_level)


def run_resample(args):
    check_files(args)
    # complex data: 2-component vector image (as written by tomio)
    as_complex = imagefile.read_info(args.input)["components"] == 2
    vol = volume.read(args.input, as_complex=as_complex)
    if args.like is not None:
        check_input(args.like)
        header = volume.read_header(args.like)
        target = dict(like=volume.Geometry(shape=header["shape"][:3], spacing=header["spacing"], origin=header["origin"], transform=header["transform"]))
    else:
        target = dict(spacing=args.spacing)
    kwargs = {} if args.method in ("labels", "roi") else dict(fill=args.fill)
    out = volume.resample(vol, method=args.method, **target, **kwargs)
    dtype = get_dtype(args.dtype)
    if dtype is not None:
        out = out.astype(dtype)
    volume.write(args.output, out, compression_level=args.compression_level)


def run_info(args):
    check_input(args.input)
    header = volume.read_header(args.input)
    info = imagefile.read_info(args.input)
    geometry = volume.Geometry(shape=header["shape"][:3], spacing=header["spacing"], origin=header["origin"], transform=header["transform"])
    lines = [
        ("file", args.input),
        ("shape", " x ".join(str(n) for n in header["shape"])),
        ("pixel type", info["pixel_type"]),
        ("spacing", format_values(geometry.spacing)),
        ("origin", format_values(geometry.origin)),
        ("orientation", f'{volume.get_orientation(geometry)} ("from" convention), {volume.get_orientation(geometry, convention="to")} ("to" convention)'),
        ("directions", "; ".join(format_values(vec) for vec in geometry.transform)),
        ("extent", format_values(geometry.extent)),
        ("metadata", f"{len(header['metadata'])} keys"),
    ]
    width = max(len(name) for name, _ in lines)
    for name, value in lines:
        print(f"{name:<{width}}  {value}")
    if args.metadata:
        for key, value in sorted(header["metadata"].items()):
            print(f"  {key} = {value}")


def run_register(args):
    from . import elastix

    check_input(args.fixed)
    for file in args.moving:
        check_input(file)
    outputs = output_names(args.output, args.moving, fixed=args.fixed)
    transforms = output_names(args.save_transform, args.moving, fixed=args.fixed) if args.save_transform else []
    figures = output_names(args.plot, args.moving, fixed=args.fixed) if args.plot else []
    check_outputs(outputs + transforms + figures, args.force)
    if args.plot:
        from . import elastix_plot  # noqa: F401  (matplotlib missing: fail before registering)

    sections = {key: getattr(args, key) for key in ("transform", "metric") if getattr(args, key)}
    config = configfile.read(args.config) if args.config else None
    stages = elastix.make_stages(config, **sections)
    fixed_mask = args.fixed_mask
    if fixed_mask in ("auto", "foreground"):
        for stage in stages:
            stage["mask"] = fixed_mask
        fixed_mask = None
    print("Registration settings:")
    for i, stage in enumerate(stages):
        if len(stages) > 1:
            print(f"stage {i + 1}:")
        print(stage)

    fixed = read_image(args.fixed)
    masks = {key: read_image(file) for key, file in [("fixed_mask", fixed_mask), ("moving_mask", args.moving_mask)] if file}
    moving = [read_image(file) for file in args.moving]
    print(f"Registering {len(moving)} image(s) onto {args.fixed}")
    registered, transformations = elastix.register(
        fixed, moving, stages, return_transform=True, reset=args.reset, loglevel="INFO" if args.verbose else "WARN",
        **masks)

    for i, (im, trans) in enumerate(zip(registered, transformations)):
        print(f"{args.moving[i]}: {trans}")
        write_image(outputs[i], im)
        if transforms:
            configfile.write(transforms[i], trans.serialize())
            print(f"  written: {transforms[i]}")
        if figures:
            from . import elastix_plot

            title = f"{Path(args.moving[i]).name} -> {Path(args.fixed).name}"
            fig = elastix_plot.plot_registration(fixed, moving[i], trans, title=title)
            fig.savefig(figures[i], dpi=110)
            print(f"  written: {figures[i]}")


def run_warp(args):
    from . import elastix

    check_input(args.transform)
    for file in args.moving:
        check_input(file)
    outputs = output_names(args.output, args.moving)
    check_outputs(outputs, args.force)
    transformation = elastix.as_transformation(configfile.read(args.transform))
    ref = None
    if args.ref is not None:
        check_input(args.ref)
        ref = read_image(args.ref)
    for file, output in zip(args.moving, outputs):
        write_image(output, elastix.transform(transformation, read_image(file), ref=ref, method=args.method))


def run_interpolate_slices(args):
    from . import elastix

    check_files(args)
    roi = volume.read(args.input)
    for label in args.drop_label:
        roi[roi == label] = 0
    image = None
    if args.image is not None:
        check_input(args.image)
        image = volume.read(args.image)
    method = args.method or ("registration" if image is not None else "shape")
    write_image(args.output, elastix.interpolate_slices(roi, image=image, method=method, axis=args.axis))


# template of `tomio register-config FILE`
TEMPLATE_CONFIG = [
    {"transform": "rigid", "metric": "mi"},
    {"transform": {"type": "bspline", "grid_spacing": 20.0, "bending": 1.0, "axis": None}, "metric": "mi"},
]


def run_register_config(args):
    from . import elastix

    if args.output is None:
        print(elastix.Config.describe())
        return
    check_outputs([args.output], args.force)
    configfile.write(args.output, TEMPLATE_CONFIG)
    print(f"written: {args.output}")


# utilities

def read_image(filename):
    """volume from a file (2-component images: complex)"""
    as_complex = imagefile.read_info(filename)["components"] == 2
    return volume.read(filename, as_complex=as_complex)


def write_image(filename, image):
    volume.write(filename, image)
    print(f"  written: {filename}")


def split_name(filename):
    """file name without directory and extension (.nii.gz: double extension), extension"""
    name = Path(filename).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7], name[-7:]
    return Path(name).stem, Path(name).suffix


def output_names(template, files, **keywords):
    """output file names from a template (<moving>, <date>, <datetime>, keywords; no extension: that of the file)"""
    now = datetime.datetime.now()
    names = []
    for file in files:
        stem, ext = split_name(file)
        name = template.replace("<moving>", stem)
        name = name.replace("<date>", now.strftime("%Y%m%d")).replace("<datetime>", now.strftime("%Y%m%d-%H%M%S"))
        for key, value in keywords.items():
            name = name.replace(f"<{key}>", split_name(value)[0])
        if not split_name(name)[1]:
            name += ext
        names.append(Path(name))
    if len(set(names)) < len(names):
        raise UserError(f"duplicate output files (use <moving> in the template): {', '.join(map(str, names))}")
    return names


def check_outputs(files, force=False):
    for file in map(Path, files):
        if not file.parent.is_dir():
            raise UserError(f"output directory not found: {file.parent}")
        if file.exists() and not force:
            raise UserError(f"output file exists (use --force to overwrite): {file}")


def check_input(filename):
    if not Path(filename).is_file():
        raise UserError(f"input file not found: {filename}")


def check_files(args):
    check_input(args.input)
    if Path(args.output).exists() and not args.force:
        raise UserError(f"output file exists (use --force to overwrite): {args.output}")


def get_dtype(name):
    if name is None:
        return None
    try:
        return np.dtype(name)
    except TypeError:
        raise UserError(f"invalid dtype: {name}") from None


def format_values(values):
    return "(" + ", ".join(f"{v:.6g}" for v in values) + ")"

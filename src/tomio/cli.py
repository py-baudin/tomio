""" command-line interface

    tomio convert IN OUT
    tomio resample IN OUT (--like REF | --spacing SX SY SZ) [--method METHOD]
    tomio info IN
"""
import sys
import argparse
from pathlib import Path

import numpy as np

from . import volume, imagefile

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
    parser = argparse.ArgumentParser(prog="tomio", description="Convert, resample and inspect medical image volumes.")
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


# utilities

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

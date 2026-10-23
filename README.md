# tomio

I/O utilities for medical images: image volumes (thin wrapper over SimpleITK), interpolation, registration, label files (ITK-SNAP, FreeSurfer, 3D Slicer) and config files (JSON, YAML, TOML).

## Installation

```bash
pip install tomio
pip install tomio[interp]   # optional: scipy interpolation backend
pip install tomio[nibabel]  # optional: conversion to/from nibabel images
pip install tomio[plot]     # optional: registration figures (matplotlib)
```

Requires Python >= 3.10. Reading TOML files on Python 3.10 requires `tomli`.

## Quick start

```python
import tomio as io

vol = io.read_volume("vol.nii.gz")  # numpy array with spacing, origin, transform, metadata
io.write_volume("vol.mha", vol)

vol2 = io.resample(vol, spacing=(1, 1, 1))

cfg = io.read_config("config.yml")
labels = io.read_labels("labels.txt")
```

Modules: `io.volume`, `io.interp`, `io.elastix` (registration), `io.labelfile`, `io.configfile`.

## Conventions

- Arrays are indexed `(x, y, z, ...)`, in ITK index order (not the `(z, y, x)` order of `sitk.GetArrayFromImage`).
- World coordinates are LPS (ITK, DICOM). Conversion to nibabel (RAS) is done by `to_nibabel`/`from_nibabel`.
- `vol.transform` is a tuple of axis directions: `transform[i]` is the world direction of array axis `i`, i.e. column `i` of the ITK direction matrix.
- Orientation codes use the ITK "from" convention by default (identity: `RAI`); the "to" convention (identity: `LPS`) is available with `convention="to"`.

## Volumes

```python
import numpy as np
import tomio as io

# read / write (formats: .nii, .nii.gz, .mha, .mhd, .nrrd...)
vol = io.read_volume("vol.nii.gz")
io.write_volume("vol2.mha", vol)

# options
vol = io.read_volume("vol.nii.gz", dtype=np.float32)  # cast
vol = io.read_volume("complex.mha", as_complex=True)  # complex volume (written by io.write_volume)
io.write_volume("vol2.nii.gz", vol, compression_level=9)
header = io.volume.read_header("vol.nii.gz")  # shape, spacing, origin, transform, metadata (no pixel data)

# Volume is a numpy ndarray
isinstance(vol, np.ndarray)  # True
vol.A  # plain np.ndarray

# header
vol.spacing  # (sx, sy, sz)
vol.origin  # world coordinates of voxel (0, 0, 0)
vol.transform  # axis directions
vol.metadata  # dict[str, str], written to the file
vol.header  # dict of spacing, origin, transform and metadata
vol.affine  # 4x4 affine transform (indices -> LPS coordinates)

# world coordinates <-> indices
coords = vol.get_coords([1, 2, 3])
indices = vol.get_indices(coords)

# create volumes
vol2 = io.Volume(np.zeros((10, 20, 30)), spacing=(1, 1, 2), origin=(0, 0, 0))
vol2 = io.asvolume(array, ref=vol)  # copy the header of `vol`
vol2 = io.tovolume(values, mask=mask, ref=vol)  # fill the voxels of a mask
```

Volumes with more than 3 dimensions (e.g. time series `(x, y, z, t)`) and complex volumes can be read and written; the geometry applies to the first 3 axes.

### Geometry

```python
geom = vol.geometry  # io.Geometry: shape, spacing, origin, transform (immutable)
geom.affine, geom.extent, geom.center, geom.voxel_volume
geom.same_as(other_vol, tol=1e-4)
vol2 = io.Volume(array, geometry=geom)
```

### Slicing and numpy operations

The geometry follows numpy operations when it makes sense:

```python
vol[2:].origin  # world coordinates of the first voxel of the slice
vol[::2].spacing  # spacing x 2 in the first axis
vol[::-1]  # axis direction flipped (also np.flip)
vol.T, np.moveaxis(vol, 0, -1)  # spacing and directions permuted
vol + 1, np.sqrt(vol)  # same geometry

# no geometry: plain np.ndarray
vol[0], vol[mask], vol.sum(axis=0), vol.mean()
# time series: geometry kept
vol4d[..., 0], vol4d.mean(axis=3)
```

### Orientation

```python
vol.anatomical_orientation  # e.g. "RAI" (ITK "from" convention)
vol.get_orientation(convention="to")  # e.g. "LPS"
vol2 = vol.reorient("RAS", convention="to")  # flip/permute the axes
```

### Interoperability

```python
image = vol.to_sitk()  # SimpleITK Image
vol = io.Volume.from_sitk(image)
nifti = vol.to_nibabel()  # nibabel Nifti1Image (requires nibabel)
vol = io.Volume.from_nibabel(nifti)
```

## Interpolation

```python
import numpy as np
import tomio as io

vol2 = io.resample(vol, like=ref)  # geometry of another volume (or a Geometry)
vol2 = io.resample(vol, spacing=(1, 1, 1))  # new spacing, same extent
vol2 = io.resample(vol, geometry={"spacing": (1, 1, 1), "shape": (64, 64, 64)})
vol2 = io.resample(vol, like=ref, method="linear", fill=np.nan)  # value outside `vol`

# label maps and masks
labels2 = io.resample(labelmap, like=ref, method="labels")  # label by label (smooth boundaries)
roi2 = io.resample(roi, like=ref, method="roi")  # no label lost when downsampling
```

- Methods: `nearest`, `linear`, `spline3` (and `splineN`), `labels`, `roi`. Default: `nearest` for integer and bool data (label maps), `spline3` otherwise. The output has the dtype of the input.
- Backends: SimpleITK (default, multithreaded) or scipy (`backend="scipy"`, requires `tomio[interp]`). SimpleITK considers voxels inside the source up to half a voxel beyond the border voxel centers.
- Lower-level functions: `io.interp.interpolate`, `interpolate_like`, `interpolate_to`, `interpolate_roi`, `interpolate_labels`.

## Label files

```python
import tomio as io

labels = io.read_labels("labels.txt")  # ITK-SNAP label description file
labels = io.read_labels("FreeSurferColorLUT.txt", format="freesurfer")
labels = io.read_labels("labels.ctbl", format="slicer")  # 3D Slicer color table
io.write_labels("labels.txt", labels)  # or labels.save("labels.txt"), ITK-SNAP format

# Labels: mapping {index: description}, sorted by index
labels = io.labelfile.Labels({1: "white matter", 2: "grey matter"})  # automatic colors
labels = io.labelfile.Labels({1: {"LABEL": "white matter", "RGBA": (255, 255, 255, 1.0)}})
labels = io.labelfile.Labels(["background", "white matter"])  # indices 0, 1

labels[1]  # description
labels[3] = "csf"  # add or rename (keeps the color)
del labels[3]
labels.getindex("grey matter")  # index of a description
labels.indices, labels.descriptions  # lists, sorted by index (can be assigned)
labels.colors  # {index: (R, G, B, A)}, RGB in 0..255, alpha in 0..1 (can be assigned)

# label maps
labels = io.labelfile.Labels.from_volume(labelmap)  # one label per value
rgba = labels.lut[labelmap]  # RGBA image (uint8)
labels2, mapping = labels.reindex()  # indices 1..N
labelmap2 = io.labelfile.apply_mapping(labelmap, mapping)

# plain data (JSON, YAML, config files): descriptions, colors and visibility
data = labels.to_dict()  # {index: {"LABEL": ..., "RGBA": [R, G, B, A], "VIS": 1, "MSH": 1}}
io.write_config("config.json", {"labels": data})
labels = io.labelfile.Labels.from_dict(io.read_config("config.json")["labels"])  # string indices accepted
```

Labels are validated: non-negative integer indices, unique descriptions without `"` or line breaks (`allow_duplicates=True` to allow duplicates), RGB in 0..255, alpha in 0..1. Label files are UTF-8.

## Registration

`tomio.elastix` (aliases in `tomio`): an Elastix-like registration framework in numpy and SimpleITK. A registration
stage combines a transform, a multi-resolution pyramid, a sampler, an interpolator, a metric and an optimizer;
transforms map fixed image points to the moving image.

```python
import tomio as io
from tomio import elastix

# rigid, mutual information (images of different contrasts)
registered, trans = io.register(fixed, moving, transform="rigid", metric="mi", return_transform=True)

# successive stages: rigid, then free-form deformation (B-spline): T(x) = T2(T1(x))
stages = [
    {"transform": "rigid", "metric": "mi"},
    {"transform": {"type": "bspline", "grid_spacing": 20, "bending": 1.0}, "metric": "mi"},
]
registered, trans = io.register(fixed, moving, stages, return_transform=True)

# apply to other images (same grid as `moving`): images, label maps, ROIs drawn on some slices
roi_fixed = io.warp(trans, roi, ref=fixed, method="roi")
labels_fixed = io.warp(trans, labelmap, ref=fixed, method="labels")

# save / load (YAML or JSON)
io.write_config("transform.yml", trans.serialize())
trans = elastix.as_transformation(io.read_config("transform.yml"))

print(elastix.Config.describe())  # all components and options
```

- Sections: `transform` (`translation`, `rigid`, `affine`, `bspline`), `pyramid`, `sampler`, `interpolator`,
  `metric` (`mse`, `mi`, `ngf`, `mind`), `optimizer` (`auto`, `lbfgs`, `sgd`), `initialize`, `resample`. Each
  section is a type name, or a dict with `type` and options; a list of settings runs successive stages.
- Same contrast: `mse`. Different contrasts: `mi` (mutual information), `ngf` (normalized gradient fields) or
  `mind` (modality independent neighbourhood descriptor).
- B-splines: `grid_spacing` (mm, final level; coarser at the coarser pyramid levels), `bending` (smoothness:
  0.1–1 with `mi`, `ngf`, `mind`), `axis=2` for in-plane deformations only (slices stay slices, e.g. for ROIs drawn
  on some slices). Run them after a converged rigid stage.
- Optimizers: `auto` uses `lbfgs` for global transforms and `sgd` (new random samples every iteration, as Elastix)
  for B-splines.
- Masks: `register(..., fixed_mask=mask)` draws samples inside the mask only.

### Deformations and sparse ROIs

```python
check = io.check_folding(trans)  # Jacobian determinant on the fixed grid
check.ok, check.min, check.max  # no folding: determinant > 0 everywhere
field = elastix.deformation_field(trans)  # displacement (mm), grid shape + (3,)
itk_transform = elastix.to_sitk(trans)  # SimpleITK transform (affine, B-spline, composite)

# ROI drawn on some slices -> all slices between them (labels interpolated as signed distances)
roi_full = io.interpolate_slices(roi)
roi_full = io.interpolate_slices(roi, image=moving, method="registration")  # guided by the image
```

`method="registration"` warps the signed distances along the image through 2D registrations of consecutive slices
(slower, more accurate across large gaps).

### Plots

`tomio.elastix_plot` (requires `tomio[plot]`, matplotlib): `plot_registration(fixed, moving, trans)`: slices of the
fixed, moving and registered images, overlays, checkerboard and, for deformations, the deformed grid;
`plot_levels(info)`: cost per pyramid level (`info` from `elastix.Elastix(config).execute(fixed, moving)`).

## Config files

```python
import tomio as io

cfg = io.read_config("config.yml")  # .json, .yaml, .yml, .toml -> dict
io.write_config("config.yml", cfg)  # .json, .yaml, .yml
```

- The format is set by the file extension (no extension: `.json`). TOML files are read-only.
- Values are written as standard types: numpy arrays as lists, numpy scalars as numbers, complex numbers as strings `"3 + 4i"` (read back as complex numbers).
- YAML files are loaded safely (no Python-specific tags).

## Command line

```bash
tomio convert vol.mha vol.nii.gz                        # convert (format from the extensions)
tomio convert vol.mha vol.nii.gz --dtype float32 --compression-level 9
tomio resample vol.nii.gz out.nii.gz --spacing 1 1 1    # new spacing
tomio resample labels.nii.gz out.nii.gz --like ref.nii.gz --method labels
tomio info vol.nii.gz --metadata                        # header: shape, pixel type, geometry

# registration (output name templates: <moving>, <fixed>, <date>, <datetime>)
tomio register-config config.yml                       # template settings (rigid, then B-spline) to edit
tomio register t1.nii.gz dixon.nii.gz -c config.yml -o "<moving>_on_<fixed>" --save-transform --plot
tomio register t1.nii.gz dixon.nii.gz --transform rigid --metric mi
tomio warp transform_dixon.yml roi.nii.gz --method roi -o roi_on_t1.nii.gz
tomio interpolate-slices roi.nii.gz roi_full.nii.gz --image dixon.nii.gz --drop-label 14
```

Existing output files are not overwritten unless `--force` is given. See `tomio COMMAND --help`.

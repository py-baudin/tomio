# tomio

I/O utilities for medical images: image volumes (thin wrapper over SimpleITK), interpolation, label files (ITK-SNAP, FreeSurfer, 3D Slicer) and config files (JSON, YAML, TOML).

## Installation

```bash
pip install tomio
pip install tomio[interp]   # optional: scipy interpolation backend
pip install tomio[nibabel]  # optional: conversion to/from nibabel images
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

Modules: `io.volume`, `io.interp`, `io.labelfile`, `io.configfile`.

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
```

Labels are validated: non-negative integer indices, unique descriptions without `"` or line breaks (`allow_duplicates=True` to allow duplicates), RGB in 0..255, alpha in 0..1. Label files are UTF-8.

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
```

Existing output files are not overwritten unless `--force` is given. See `tomio COMMAND --help`.

# Changelog

## Unreleased

### Added
- Command-line interface `iomed` (also `python -m iomed`):
  - `iomed convert IN OUT [--dtype DTYPE] [--compression-level N]`: convert image files (format from the extensions), keeping geometry, metadata, vector and complex pixels.
  - `iomed resample IN OUT (--like REF | --spacing SX SY SZ) [--method nearest|linear|spline3|labels|roi] [--fill VALUE]`.
  - `iomed info IN [--metadata]`: shape, pixel type and geometry (header only).
  - Existing output files are not overwritten without `--force`; errors are reported without traceback (exit code 1; usage errors: 2).
- `imagefile.convert_image` and `imagefile.read_info` (pixel type, number of components, dimension).
- `configfile.read` reads TOML files (`.toml`, read-only), with `tomllib` (Python >= 3.11) or `tomli`.
- Labels:
  - `Labels(["a", "b"])`: from a sequence of descriptions (indices 0, 1...); `indices=` and `colors=` arguments.
  - Setters `labels.indices`, `labels.descriptions`, `labels.colors` (sequence in index order; `colors` also accepts a dict): validated like `set()`, labels unchanged on error.
  - `labels.getindex(name)` returns the first index with that name (when duplicates are allowed).
  - `labels.lut`: RGBA lookup table (`uint8`, shape `(max index + 1, 4)`), e.g. `rgba = labels.lut[labelmap]`.
  - `Labels.from_volume(labelmap, names=None)`: labels from the values of a label map, with automatic colors.
  - `labels.reindex()`: consecutive indices 1..N, returns the new labels and the mapping `{old: new}`; `labelfile.apply_mapping(labelmap, mapping)` (also `Labels.apply_mapping`) applies it to a label map.
  - `labelfile.read(filename, format="freesurfer" | "slicer")`: FreeSurfer (`FreeSurferColorLUT.txt`) and 3D Slicer (`.ctbl`) color tables.
- Interoperability: `vol.to_sitk()` / `Volume.from_sitk(image)` (SimpleITK `Image`), and `vol.to_nibabel()` / `Volume.from_nibabel(image)` (nibabel `Nifti1Image`, LPS ↔ RAS conversion of the affine; optional dependency: `pip install iomed[nibabel]`).
- Orientation conventions: `vol.get_orientation(convention="from" | "to")` and `vol.reorient(code, convention=...)`. "from" (default, ITK, identity: `RAI`) is the convention of `vol.anatomical_orientation`; "to" (identity: `LPS`) gives the direction each axis points to. `volume.convert_orientation` converts codes between them.
- `volume.Geometry` (also `iomed.Geometry`): immutable geometry (shape, spacing, origin, transform) with `affine`, `extent`, `center`, `voxel_volume`, `get_coords`, `get_indices`, `same_as(other, tol)`, `to_dict`/`from_dict`. `vol.geometry` returns it (first 3 axes), `Volume(data, geometry=...)` accepts it, and interpolation functions accept it in place of geometry dicts.
- `resample(vol, like=ref | spacing=... | geometry={...}, method=..., fill=...)`: single entry point for interpolation, in `interp`, `volume` and `iomed`. Methods: `nearest`, `linear`, `splineN`, `roi`, `labels`. The `interpolate*` functions are unchanged.
- `volume.read_header(filename)`: shape, geometry and metadata, without reading the pixel data.
- `volume.read(..., dtype=...)`: cast the data.
- `volume.write(..., compression_level=...)`: compression level (e.g. 1-9 for `.nii.gz`).

- `interp.interpolate_roi(..., method="spline1")` (or any interpolation method): higher quality upsampling of label maps. Each label is interpolated separately as a binary mask, and each voxel gets the label with the highest value. Also available as `interp.interpolate_labels`.

### Changed
- Interpolation backends (`backend=` argument of the interpolation functions and `resample`):
  - `"sitk"` (default): SimpleITK, multithreaded (~1.5-2x faster than scipy), no scipy needed. Voxels are inside the source up to half a voxel beyond the border voxel centers (scipy: up to the border voxel centers), so border voxels can differ from previous versions; inside the volume, results are identical.
  - `"scipy"`: `scipy.ndimage.affine_transform` (was: `map_coordinates` with all target coordinates, ~90 bytes per target voxel); single-slice sources use `map_coordinates`.
  - Complex data: real and imaginary parts are interpolated separately (supported by both backends); the fill value applies to the real part.
  - Default backend: scipy is used automatically (if installed) for what SimpleITK does not support: modes `"mirror"`, `"wrap"`, and `disjoint=True` with `interpolate_roi(..., method=...)`.
- Require Python >= 3.10 and numpy >= 1.24.
- scipy is now an optional extra: `pip install iomed[interp]` (also included in the `test` extra); interpolation works without it (SimpleITK backend).
- `iomed.interp` is available as an attribute of the package.

- `Labels` is a full mutable mapping (`update`, `setdefault`, `clear`, `popitem`...); `Labels()` creates an empty object and any mapping is accepted as input.
- `labels.save(filename)` works on an instance (it raised `TypeError`); `Labels.save(filename, labels)` still works.
- `Labels` can be compared with a plain mapping (descriptions only), e.g. `labels == {1: "a"}`.

### Fixed
- Writing boolean volumes no longer uses the removed `np.bool` alias.
- `interp.interpolate_to` no longer crashes (`KeyError: 'shape'`).
- Interpolation no longer replaces NaNs with 0 in the input arrays.
- `volume.write(nan_as=...)` no longer replaces NaNs in the input array.
- `volume.write` supports complex volumes (it crashed): they are written as 2-component vector images (real, imaginary), read back with `volume.read(..., as_complex=True)`.
- `volume.write`/`read` support volumes with more than 3 dimensions, e.g. time series (x, y, z, t) (writing crashed). The geometry applies to the first 3 axes.
- `volume.write` writes `vol.metadata` to the file (it was ignored). Which keys are kept depends on the format: MetaImage (`.mha`) keeps all keys, NIfTI only its own header fields.
- Label descriptions can contain any character except `"` and line breaks (which are now rejected), and are read back correctly.
- `Labels.pop(index, default)` accepts a default value, like `dict.pop`.
- `Labels`: `indices`, `keys()` and `values()` are sorted by index, like iteration and `descriptions` (they could be out of sync after an update).
- `Labels(other)` makes an independent copy (modifying it changed `other`), and keeps `allow_duplicates`.
- `Labels`: an invalid update (bad or duplicate description...) no longer deletes the existing label.
- Label visibility (VIS, MSH) is read from label files (was reset to 1).
- `config.read` accepts `.yaml` files (and upper-case extensions).
- `config.tame` and config files: booleans stay booleans (`True` became `1`, `np.True_` became `"True"`), complex numbers are saved as `"3 + 4i"` (writing them crashed), 0-d arrays become scalars.
- Label files: errors give the file name and line number (`labelfile.LabelFileError`, a subclass of `ValueError` and `RuntimeError`); duplicate indices are rejected; alpha accepts any decimal format (`.5`, `1.`, `1e-1`).
- `Labels.palette` raises an error for indices > 255 (they were silently dropped).

### Changed (breaking)
- `Labels.pop(index)` returns the label description (was: the full info dict), consistent with `labels[index]`.
- `Labels` validates its input: indices must be non-negative integers (numpy integers accepted), RGB values integers in 0..255, alpha in 0..1, a description (LABEL) is required, and unknown fields are rejected.
- `Labels(...)` and `Labels.save(...)` raise `TypeError` (was `ValueError`) when the input is not a mapping. `Labels.re_parse_label` is removed.
- YAML config files are loaded safely (`yaml.safe_load`): Python-specific tags (`!!python/...`) are rejected.
- Config files: strings of the form `"3 + 4i"` are read as complex numbers.
- Geometry-aware slicing of `Volume`: slices on the spatial axes update origin and spacing (e.g. `vol[2:]`, `vol[::2]`), and negative steps flip the axis direction (`vol[::-1]`, `np.flip(vol)`). Other indexing on the spatial axes returns a plain `np.ndarray`: integer index (`vol[0]` was a 2D `Volume` with the 3D header), arrays and masks (`vol[mask]`), `None`. Integer indices on the non-spatial axes (e.g. `vol[..., 0]` for a time series) keep the geometry.
- `Volume` reductions (`sum`, `mean`, `max`, `std`, `argmax`, `argmin`...) over the spatial axes (or all axes) return a plain `np.ndarray` (or scalar); reductions over the non-spatial axes only (e.g. `vol.mean(axis=3)`) keep the geometry.
- `Volume` axis permutations (`.T`, `transpose`, `swapaxes`, `np.moveaxis`...) permute spacing and transform (they were unchanged); a plain `np.ndarray` is returned if a spatial axis is moved beyond the first 3 axes.
- `Volume` elementwise operations take the header of the first `Volume` operand with the shape of the result (e.g. `vol[:1] + vol` has the header of `vol`); when the shape changes (`np.matmul`, `outer`), the result is a plain `np.ndarray`.
- Interpolation: the default method is `nearest` for integer and bool data (label maps; `spline3` created labels that do not exist), `spline3` for other data. The output has the dtype of the input (bool input is now supported).
- `volume.read` returned the transposed direction matrix (only visible with non-symmetric directions, e.g. oblique or permuted axes): `vol.transform`, `vol.affine` and `vol.get_coords` were wrong. Code that compensated for it must be updated.
- Module `config` renamed `configfile` (consistent with `labelfile` and `imagefile`). `iomed.config` and `from iomed import config` still work (alias); `import iomed.config` does not.
- Internal module `iovolume` renamed `imagefile`, and `iovolume.write_volume` renamed `imagefile.write_image` (`read_image` unchanged).
- `volume.read` of vector images (e.g. RGB, vector fields): the components are in the last axis (x, y, z, c); they were in the first axis (c, x, y, z).
- `config.write` keeps the given file extension (`c.yaml` was written as `c.yml`); without extension, `kind` is used as extension (`kind="yaml"` gives `.yaml`, was `.yml`).
- Config files are read and written as UTF-8.
- Importing iomed no longer registers the `!ndarray` tag in PyYAML's default dumper/loader.
- Label files are read and written as UTF-8 (was: the system default encoding). Files with non-ASCII labels saved by earlier versions on Windows must be re-encoded.
# Changelog

## Unreleased

### Changed
- Require Python >= 3.10 and numpy >= 1.24.
- scipy is now an optional extra: `pip install iomed[interp]` (also included in the `test` extra).
- `iomed.interp` is available as an attribute of the package (imported lazily).

- `Labels` is a full mutable mapping (`update`, `setdefault`, `clear`, `popitem`...); `Labels()` creates an empty object and any mapping is accepted as input.
- `labels.save(filename)` works on an instance (it raised `TypeError`); `Labels.save(filename, labels)` still works.
- `Labels` can be compared with a plain mapping (descriptions only), e.g. `labels == {1: "a"}`.

### Fixed
- Writing boolean volumes no longer uses the removed `np.bool` alias.
- `interp.interpolate_to` no longer crashes (`KeyError: 'shape'`).
- Interpolation no longer replaces NaNs with 0 in the input arrays.
- `volume.write(nan_as=...)` no longer replaces NaNs in the input array.
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
- `volume.read` returned the transposed direction matrix (only visible with non-symmetric directions, e.g. oblique or permuted axes): `vol.transform`, `vol.affine` and `vol.get_coords` were wrong. Code that compensated for it must be updated.
- `config.write` keeps the given file extension (`c.yaml` was written as `c.yml`); without extension, `kind` is used as extension (`kind="yaml"` gives `.yaml`, was `.yml`).
- Config files are read and written as UTF-8.
- Importing iomed no longer registers the `!ndarray` tag in PyYAML's default dumper/loader.
- Label files are read and written as UTF-8 (was: the system default encoding). Files with non-ASCII labels saved by earlier versions on Windows must be re-encoded.
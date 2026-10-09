# Changelog

## Unreleased

### Changed
- Require Python >= 3.10 and numpy >= 1.24.
- scipy is now an optional extra: `pip install iomed[interp]` (also included in the `test` extra).
- `iomed.interp` is available as an attribute of the package (imported lazily).

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

### Changed (breaking)
- `Labels.pop(index)` returns the label description (was: the full info dict), consistent with `labels[index]`.
- Label files are read and written as UTF-8 (was: the system default encoding). Files with non-ASCII labels saved by earlier versions on Windows must be re-encoded.
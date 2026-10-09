# Changelog

## Unreleased

### Changed
- Require Python >= 3.10 and numpy >= 1.24.
- scipy is now an optional extra: `pip install iomed[interp]` (also included in the `test` extra).
- `iomed.interp` is available as an attribute of the package (imported lazily).

### Fixed
- Writing boolean volumes no longer uses the removed `np.bool` alias.

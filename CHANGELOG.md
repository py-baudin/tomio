# Changelog

## Unreleased

### Added

- `elastix.Geometry.to_volume`, `from_volume`: conversion to/from `volume.Geometry`.
- README section "Registration".
- Optimizer `"asgd"`: Elastix's adaptive SGD with automatic parameter estimation.
- Sampler `"grid"` (as Elastix's ImageGridSampler); sampler option `seed`.
- MSE option `normalize` (as Elastix's UseNormalization, off by default).
- Section `mask` (`"auto"`, `"foreground"`) and `foreground_mask`; `--fixed-mask auto`.

### Changed

- `tomio.elastix` split into a subpackage (same `from tomio import elastix` names).
- Masks eroded at each pyramid level as Elastix's ErodeMask (option `erode_mask`).
- Pyramid masks are no longer dilated when downsampling.
- Default B-spline `bending`: 1 with MI, NGF, MIND; 0 with MSE (was 0).
- Default B-spline `grid_spacing` 10 mm (was 20 mm, as Elastix).

### Removed

- SGD option `adaptive` (not Elastix's method; use `"asgd"`).

## 0.2.0

Registration module `tomio.elastix`, imported from `mutools.registration.elastix` (not backward compatible).

### Added

- `tomio.elastix`: Elastix-like registration (numpy + SimpleITK), with its test suite.
- Configuration sections (transform, pyramid, sampler, ..., resample); `Config.describe()`.
- Successive stages (list of configs), returned as a `TransformationSequence`.
- Transform `"bspline"` (free-form deformation): grid schedule, `bending`, `axis`.
- Metrics `"ngf"` (normalized gradient fields) and `"mind"` (Heinrich 2012).
- MI option `limits` (intensity range from percentiles).
- Optimizers `"sgd"` (new samples every iteration) and `"auto"`; SGD option `adaptive`.
- Section `initialize` (`None`, `"geometry"`, `"moments"`).
- Pyramid option `downsample` (False: smoothing only).
- `transform(..., method="roi" | "labels")` for any transform (`warp_roi`, `warp_labels`).
- `interpolate_slices`: sparse ROI interpolation (shape or registration based).
- `spatial_jacobian`, `jacobian_determinant`, `check_folding`, `deformation_field`.
- `to_sitk`: SimpleITK equivalent of a transformation.
- `tomio.elastix_plot` (optional extra `tomio[plot]`).
- Top-level aliases `io.register`, `io.warp`, `io.interpolate_slices`, `io.check_folding`.
- CLI commands `register`, `warp`, `interpolate-slices`, `register-config`.
- Benchmarks: `bench_elastix.py`, `phantom_elastix.py`, `bench_elastix_ffd.py`.

### Changed

- Results are tomio `Volume`s (3D) or `Image`s (2D).
- 3D rigid transform: 6 Euler parameters, `R = Rz Rx Ry` (as ITK `Euler3DTransform`).
- Images have a direction matrix; `Geometry.center` is the pixel grid center.
- Rotation center fixed on the full resolution fixed image, stored in `Transformation`.
- Parameters scaled to ~mm (rotations x image radius) for the optimizers.
- Default optimizer `"auto"`: `lbfgs` (numpy, `xtol`) for global transforms, `sgd` otherwise.
- Constant sample set per level; samples leaving the moving image are padded.
- Pyramid: isotropic reduced voxels; B-spline levels >= 32 voxels per axis.
- `SamplerRandomCoordinates`: linear order, samples on the grid along thick axes.
- MI by scatter-add; B-spline weights in closed form, cached.
- B-spline `bending` relative to (100 mm)^2, independent of the grid spacing.
- Non-global transforms resampled by SimpleITK through a displacement field.
- Serialized transformations: lowercase keys.

### Fixed

- Gradients: linear interpolation spacing, MI normalization, B-spline interpolator.
- `register(..., reset=False)` reuses the parameters of the previous image.
- `transform(..., ref=...)`; serialization of transforms with `init` or `axis`.
- B-spline displacement frozen in the last control cell of each axis.
- `warp_roi` and `resample(method="roi")` keep whole slices (tilted geometries too).
- `resample(method="roi")` rounding of negative coordinates.

### Removed

- scipy dependency; `scipy` and `agd` optimizers.
- Flat settings (`Config.SETTINGS`), `interpolation_order`, `Elastix.apply`.
- 7-parameter 3D rigid transform.

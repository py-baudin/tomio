# Changelog

## Unreleased

- Added `tomio.elastix`: registration framework imported from mutools (`mutools.registration.elastix`), with its test suite.

### tomio.elastix: free-form deformation (phase 5 of `PLAN_elastix.md`, in progress)

- New transform `"bspline"` (`TransformBSpline`): cubic B-spline displacement on a control grid fitted to the
  fixed image, grid spacing schedule across the pyramid levels (`schedule`), bending energy penalty (`bending`).
- New metric `"ngf"` (normalized gradient fields), multimodal.
- `transform(..., method="roi" | "labels")` works for any transform (`warp_roi`, `warp_labels`).
- The cost gradient uses the image gradient at the moving points and the transposed transform Jacobian
  (sparse for B-splines).
- Defaults: `SamplerRandomCoordinates(order=1)`, `OptimizerLBFGS(maxiter=500)`.
- Added: elastix optimizer "sgd" (stochastic gradient descent, new samples every iteration).
- Known limitation: on real data (dixon → T1), the B-spline stage does not yet improve on the rigid result
  (see `TODO.md`).

### tomio.elastix: accelerations (phase 4 of `PLAN_elastix.md`)

- Optimized parameters are scaled (rotations and matrix coefficients × the radius of the fixed image):
  better conditioned optimization (translations were barely optimized before).
- `lbfgs`: new `xtol` stopping criterion (default 1e-3, ≈ mm), `ftol` 1e-7: far fewer cost evaluations.
- Mutual information by scatter-add (same result, faster).
- Pyramid: linear downsampling by default (`order=1`), recursive gaussian smoothing.
- `SamplerRandomCoordinates(thick_ratio=2.0)`: samples stay on the grid along thick-slice axes.
- New `initialize` section: `None`, `"geometry"` or `"moments"` (initial translation of the first stage).
- `Elastix.execute` returns per-level results in `info.levels`.
- `benchmarks/bench_elastix.py`.

### tomio.elastix: no scipy (phase 3 of `PLAN_elastix.md`)

- scipy is no longer used: B-spline coefficients with SimpleITK (`sitk.BSplineDecomposition`, cached per image),
  interpolation at points in numpy (values and analytic gradients), pyramid smoothing / resampling / mask
  dilation with SimpleITK.
- New default optimizer `"lbfgs"` (numpy L-BFGS); removed the `scipy` and `agd` optimizers.
- Removed: `parzen1d`, `Image.summary`, `Objective.gradient`, `Objective.callback`, `Elastix.apply`
  (use `transform`).

### tomio.elastix: configuration API (phase 2 of `PLAN_elastix.md`)

- New configuration: one section per component (`transform`, `pyramid`, `sampler`, `interpolator`, `metric`,
  `optimizer`, `resample`), with short type names (`"rigid"`, `"mi"`, ...), options validated against the
  components; `Config.describe()` lists the types and options; configs from YAML/JSON files.
  Replaces the flat `{"Transform": "TransformRigid", "TransformRigid": {...}}` settings and `Config.SETTINGS`.
- `register(fixed, moving, config=None, **sections)`: keyword sections (`transform="rigid", metric="mi"`),
  `fixed_mask` / `moving_mask`; a list of configs runs successive stages (`TransformationSequence`).
- `transform(transformations, moving, ref=None, method=None)`: `method` as in `tomio.resample`
  (`'nearest'`, `'linear'`, `'spline3'`, `'roi'`, `'labels'`), replaces `interpolation_order`; global
  transforms of 3d images are resampled with `tomio.resample`.
- `register` and `transform` return tomio `Volume`s for 3d images (with the geometry of the fixed / reference
  image), `Image`s for 2d images.
- Serialized transformations: lowercase keys (`transform: {type: rigid, ...}`, `geometry`, `parameters`, `center`);
  `Transformation(transform, geometry, parameters, center=None)` takes a transform object, name or dict.

### tomio.elastix: fixes and changes (phase 1 of `PLAN_elastix.md`)

Not backward compatible with `mutools.registration.elastix`:

- 3D rigid transform: 6 parameters, Euler angles and translation `(ax, ay, az, bx, by, bz)` with `R = Rz @ Rx @ Ry`
  (as ITK's `Euler3DTransform`). The 7-parameter `(phi, n, b)` form is removed (it was not rigid for `|n| != 1`).
- `Geometry.center` is the center of the pixel grid (`origin + (shape - 1) / 2 * spacing`), half a pixel off before.
  The rotation center is computed once on the full resolution fixed image and stored in `Transformation`
  (`center`, serialized as `"Center"`); `transform(ref=...)` keeps it.
- Images have a direction matrix (`Geometry.direction`, `Image.direction`, read from `direction` or from tomio's
  `transform`; `Image.transform` in tomio's convention); serialized geometries include `"direction"` when oblique.
- `PyramidGaussian`: new defaults (`scale_ratio=2`, `maxlevel=3`, `minsize=8`, `sigma=None`: 0.5 × reduction factor);
  `minsize` is now the minimum size of each reduced axis; reduced grids have the exact spacing and the same center.
- Samples outside the moving image (pixel centers ± 0.5 pixel) are discarded instead of counting as 0;
  `InterpolatorLinear` and `InterpolatorBSpline` extend edge values by default (`mode="nearest"`); the final
  resampling still fills with 0.
- `OptimizerGeneric` keeps the samples fixed during the optimization (`resample=True` for the previous behaviour).
- `Transform.deserialize` no longer converts string settings to `None`.

Fixes:

- `register(..., reset=False)` reuses the parameters of the previous moving image (`init` was ignored).
- `transform(..., ref=...)` works (it raised `AttributeError`).
- `serialize()` of rigid/affine transforms with `init`, and of translations with `axis`.
- Gradients: linear interpolation gradient w.r.t. physical coordinates (missing `1/spacing`), MI gradient with
  the intensity normalization, B-spline interpolator gradient (wrong shape, O(N²) memory), linear interpolation
  gradient outside the grid.
- `SamplerRandomCoordinates` samples the whole image (no 10 % border) and uses the mask.
- `agd` reverts rejected steps; returns the best point.
- `Image(image)` keeps the mask of `Image` inputs; `Image.update` no longer modifies the input array;
  `np.asarray(image, dtype=...)` works; intensity range computed on the masked finite values.
- MI: no masked arrays; zero-order Parzen window counts values on bin boundaries once.
- `loglevel` applies to all the module's log messages.

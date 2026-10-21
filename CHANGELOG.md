# Changelog

## Unreleased

- Added `tomio.elastix`: registration framework imported from mutools (`mutools.registration.elastix`), with its test suite.

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

# TODO

Remaining work and future features. Details, measurements and history: `PLAN_elastix.md` (local, not tracked).

## tomio.elastix

### Free-form deformation (in progress)

- [ ] **Real data**: on dixon → T1, rigid (MI) + B-spline (MI) does not improve on rigid alone (SimpleITK Mattes
  MI score -0.301 vs -0.312), whereas SimpleElastix does (-0.337). Check the results visually first.
- [ ] Find the cause: stochastic optimization with new samples at each iteration (as Elastix's ASGD), Mattes MI
  details (intensity ranges, padding bins), number of samples vs number of parameters.
- [ ] NGF metric: drifts by a near-uniform translation on real data (not penalized by the bending energy);
  gradients of thick-slice images.
- [ ] Runtime: 40–80 s for rigid + B-spline on dixon → T1 (SimpleElastix: 22 s).
- [x] MIND metric (Heinrich et al., Medical Image Analysis 2012): done (`"mind"`).
- [x] `deformation_field()`, Jacobian determinant, `check_folding()`: done (real data, bending 1: no folding,
  determinant 0.94–1.05).
- [ ] Analytic B-spline spatial Jacobian (finite differences: 44 s on the 2.9 M T1 voxels).
- [ ] Export to `sitk.BSplineTransform` (fast resampling).
- [ ] ROI warping through B-spline transforms (`warp_roi`): a z deformation spreads a labelled slice onto several
  output slices; keep the whole-slice logic of `tomio.resample(method="roi")` (one labelled slice per output slice).
  Simple fix available: in-plane deformation only (`{"type": "bspline", "axis": 2}`); general case still to do.

### Other

- [ ] Optional mask erosion by the interpolation support (as Elastix's `ErodeMask`).
- [ ] Optional `numba` kernels (interpolation, histograms).
- [ ] Gradient-based metrics: pull back the moving gradients through the transform (rotations).

## Interpolation

- [x] Smart interpolation of sparse ROIs: `elastix.interpolate_slices` (method "shape": signed distances;
  "registration": guided by 2D registrations of consecutive slices, Ogier et al. 2017). Leave-one-slice-out on
  dixon_0033 (40 mm gaps): Dice 0.62 nearest slice, 0.71 shape, 0.77 registration.
- [ ] Faster registration method (≈ 1.7 s per slice pair); morphological contour interpolation (Albu 2008) for
  topology changes; image-guided refinement (edges) of the fused labels.

## Command line interface

- [ ] Import the mutools `utils/elastix` CLI into `tomio` (argparse subcommands `register`, `transform`, show
  config, demo); output name templates, `--force`, YAML transform files, optional matplotlib plots.
- [ ] Demo images (`legs`, `brain`, `dwi`, 2 MB): copy into tomio or download.
- [ ] README section for `tomio.elastix`.

## Future features

- [ ] GPU backend (torch or cupy).
- [ ] Top-level aliases in `tomio/__init__.py` (e.g. `io.register`).

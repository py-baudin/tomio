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
- [x] Adaptive step size (ASGD time adaptation): option `adaptive`, less accurate than the fixed decay on the
  synthetic benchmark (MI 1.22 vs 1.14 mm, MSE 0.52 vs 0.37 mm): kept off.
- [ ] Finer grids now converge (10 mm: MI 1.11, MSE 0.33 mm vs 20 mm: 1.14, 0.37): check on real data before
  changing the default grid spacing (20 mm).
- [x] MIND metric (Heinrich et al., Medical Image Analysis 2012): done (`"mind"`).
- [x] `deformation_field()`, Jacobian determinant, `check_folding()`: done (real data, bending 1: no folding,
  determinant 0.94–1.05).
- [x] Analytic spatial Jacobian (T1 folding check 44 → 16 s), `to_sitk` export, SimpleITK resampling of
  non-global transforms (labels 46 → 6 s). Remaining cost: mapping the points (≈ 6 s for 2.9 M points).
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
- [x] Faster registration method (0.3 s per slice pair, same Dice). Threads do not help (GIL).
- [ ] Morphological contour interpolation (Albu 2008) for topology changes; image-guided refinement (edges) of the fused labels.

## Command line interface

- [ ] Import the mutools `utils/elastix` CLI into `tomio` (argparse subcommands `register`, `transform`, show
  config, demo); output name templates, `--force`, YAML transform files, optional matplotlib plots.
- [ ] Demo images (`legs`, `brain`, `dwi`, 2 MB): copy into tomio or download.
- [ ] README section for `tomio.elastix`.

## Future features

- [ ] GPU backend (torch or cupy).
- [x] Top-level aliases: `io.register`, `io.warp`, `io.interpolate_slices`, `io.check_folding`.

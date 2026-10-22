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
- [ ] MIND metric (Heinrich et al., Medical Image Analysis 2012): modality independent neighbourhood descriptor,
  as a multi-channel image per pyramid level, compared by mean squared difference.
- [ ] Export to `sitk.BSplineTransform` (fast resampling), `deformation_field()`, Jacobian determinant (folding check).

### Other

- [ ] Optional mask erosion by the interpolation support (as Elastix's `ErodeMask`).
- [ ] Optional `numba` kernels (interpolation, histograms).
- [ ] Gradient-based metrics: pull back the moving gradients through the transform (rotations).

## Command line interface

- [ ] Import the mutools `utils/elastix` CLI into `tomio` (argparse subcommands `register`, `transform`, show
  config, demo); output name templates, `--force`, YAML transform files, optional matplotlib plots.
- [ ] Demo images (`legs`, `brain`, `dwi`, 2 MB): copy into tomio or download.
- [ ] README section for `tomio.elastix`.

## Future features

- [ ] GPU backend (torch or cupy).
- [ ] Top-level aliases in `tomio/__init__.py` (e.g. `io.register`).

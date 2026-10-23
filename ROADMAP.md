# tomio.elastix roadmap

Next steps for the registration module, in priority order. This file replaces `TODO.md` and the earlier working
plans. Read it fully before starting a phase.

## 1. Status

- `main` = v0.2.0 (`843650d`). Branch `elastix-next` adds, on top of it:
  - the README "Registration" section and the CHANGELOG 0.2.0 section;
  - the split of `tomio.elastix` into a subpackage (same public names);
  - Elastix mask erosion, the `"asgd"` optimizer, the `"grid"` sampler, and new defaults (bending per metric,
    10 mm grid).
- Tests: `pytest -q` passes (443 passed, 9 skipped).
- Synthetic benchmark (`benchmarks/bench_elastix_ffd.py`, known deformation, error without registration: 1.71 mm).
  Mean error with the default settings:

  | case | v0.2.0 | `elastix-next` |
  |---|---|---|
  | a: MI, cross-contrast | 1.68 mm | 1.11 mm |
  | b: MSE, same contrast | 0.37 mm | 0.33 mm |
  | c: NGF | 1.29 mm | 1.12 mm |

- Not shown yet: that the B-spline stage beats rigid on real data (Dixon → T1/T2). No ground truth so far.
- Speed: rigid + B-spline takes 40–80 s on Dixon → T1, against 22 s for SimpleElastix.

## 2. Working rules

1. **Ask before committing**, and before any task marked **[ASK]**. Work on `elastix-next`. No PRs unless asked.
2. **Check what Elastix does first** (source: SuperElastix/elastix on GitHub). Adopt something else only when it is
   clearly better on the measurements; otherwise use the most consistently better setting.
3. **Measure every change** to an optimizer, sampler, metric, transform or default on benchmark cases a–c, and on
   the real-data harness (phase A) once it exists. Report before/after tables.
4. **The numpy implementation is the reference.** Any numba kernel or SimpleITK shortcut must match it in a test.
5. `pytest -q` passes before any commit proposal. Tests write only to `tmp_path`; never modify user data.
6. **CHANGELOG:** one line per entry (≤ 80 characters) under Added, Changed, Fixed or Removed in `Unreleased`.
   **Commit messages:** imperative, ≤ 72 characters, with a type prefix (`feat`, `fix`, `perf`, `docs`,
   `refactor`, `test`, `chore`) and an optional scope.
7. Files: write them with `encoding="utf-8"`. Run edit scripts from files, not from shell heredocs with nested
   quotes.

## 3. Phases

### A. Evaluation with ground truth (priority)
- **A1. [ASK] Data.** Choose one or more:
  - ROIs drawn on a few T1/T2 slices of the Dixon case (scored by Dice of the warped Dixon ROI);
  - landmarks;
  - an in-phase or water Dixon, which avoids the out-of-phase cancellation rims;
  - a public set: Learn2Reg 2021 CT–MR thorax-abdomen, multimodal, same patient, organ labels (needs a
    grand-challenge account).
- **A2. Harness.** `benchmarks/realdata_elastix.py FIXED MOVING [--fixed-roi R] [--moving-roi R] [--config C ...]`.
  One row per configuration:
  - Dice per label (ROIs warped with `method="labels"`);
  - held-out metric inside the fixed mask;
  - an independent SimpleITK Mattes MI score;
  - folding (`check_folding`), runtime and number of parameters.

  Add an optional SimpleElastix / itk-elastix row (its default B-spline map) when installed. Data paths are
  arguments; data is never committed.
- **A3. Parity with SimpleElastix.** Table of its parameter map against the tomio settings: resolutions, pyramid,
  grid spacing and schedule, iterations, samples, bins, interpolators, bending, masks. Change one difference at a
  time and report the effect.
- **A4. Defaults on real data.** Confirm or revise the choices made on the synthetic phantom only:
  - bending 1 with MI/NGF/MIND;
  - the 10 mm grid with the ×2 schedule;
  - SGD vs ASGD;
  - the automatic mask (`mask="auto"`);
  - MI vs MIND for the out-of-phase Dixon.

**Done when:** on real data, the B-spline stage improves Dice over rigid alone (or both independent scores when
there are no ROIs), or the parity table points to the remaining difference.

### B. Monitoring
- Held-out metric every `monitor` iterations (default 25) and at the end of each level, on a fixed held-out set
  (10 × `maxsample`, inside the mask). Store it in `info.levels[k].history`.
- Warn when the optimized cost improves while the held-out metric worsens over the last third of a level.
- Held-out curves in `elastix_plot.plot_levels`.

### C. Metrics
- **C1. MIND in physical units.** Offsets and `sigma` in mm, along the fixed image axes. The moving descriptors use
  the same physical offsets, so that grids with different spacings or orientations agree. Test: the same image on
  two grids gives the same descriptors.
- **C2. NGF.**
  - Pull the moving gradients back through the transform (`J^T ∇M(T(x))`), needed for rotations.
  - Option `TransformBSpline(remove_mean=True)` against the near-uniform drift seen on real data.
  - Tests against finite differences.
- **C3. Weighted multi-metric.** `metric=[{"type": "mi", "weight": 1}, {"type": "ngf", "weight": 0.5}]`, with shared
  samples (as Elastix's multi-metric registration). Benchmark MI + NGF and MI + MIND.
- **C4. Sample checks.** Log the fraction of samples mapped outside the moving image per level; warn above 25 %.

### D. Speed
- Profile one B-spline level on real data and include the cProfile summary.
- Mapping the points of large grids still takes about 6 s for 2.9 M points.
- Optional numba kernels behind `backend="numpy" | "numba"` (extra `tomio[numba]`): B-spline displacement, moving
  interpolation and gradient, metric contributions, scatter of `J^T g`. Must match numpy (rtol 1e-10 for MSE,
  1e-8 for the others).
- `from_sitk`, and reading/writing `.tfm`/`.h5` transforms through SimpleITK.
- Target: rigid + B-spline on Dixon → T1 within 2× of SimpleElastix's 22 s on the same machine.

### E. ROI warping through z deformations
- `warp_roi` for non-global transforms: warp each label's signed distance map (shared with `interpolate_slices`),
  take the most negative label, then apply the whole-slice rule (one output slice per labelled slice).
- Tests: a sparse ROI warped by a deformation with a z component stays on whole slices and keeps its labels.
- Until then: in-plane B-splines (`axis=2`), or fill the ROI with `interpolate_slices` first (`interp_fill.py`).

### F. Sparse ROI interpolation
- Topology changes (distal slices, where Dice is lowest, 0.33–0.49): **[ASK]** morphological contour interpolation
  (Albu 2008, ITK module, adds an `itk` dependency) or a numpy equivalent.
- Image-guided refinement of the fused labels, e.g. along edges.

### G. Sampling
- `SamplerRandomCoordinates(mode="stratified")`: jittered samples on a regular grid. Make it the default only if it
  is at least as accurate (compare gradient variance and benchmark accuracy).
- Optional importance sampling from the image gradients, with weights 1/(N p) in every metric.

### H. CLI
- Check `register`, `warp` and `interpolate-slices` on 2D images and single-slice volumes.
- **[ASK]** `demo` command (mutools' iteration plots on the `legs`, `brain` and `dwi` images, 2 MB): keep or drop.

### I. Release
- Merge `elastix-next` into `main`, bump the version, and move `Unreleased` to a version section in the CHANGELOG.
- **[ASK]** Rename the module (e.g. `tomio.registration`, keeping `tomio.elastix` as an alias)? Add an importer for a
  subset of Elastix parameter files (it would help A3)?

### Optional
- Symmetric registration (half-way space, inverse consistency).
- Out of scope: GPU backends, deep-learning features, stationary velocity fields, DICOM SEG/RTSTRUCT.

## 4. Findings (don't rediscover)

- **Conventions:**
  - arrays are indexed `(x, y, z)`; world coordinates are LPS;
  - tomio's `transform` holds the columns of the ITK direction matrix (adapter: `Geometry.to_volume` /
    `from_volume`);
  - transforms map fixed → moving, and stages compose as T_n(…T_1(x));
  - the rotation center is fixed once, at the full-resolution fixed image center;
  - 3D rigid transforms use Euler angles R = Rz·Rx·Ry (ITK);
  - parameters are scaled so that one unit ≈ 1 mm.
- **Samples:** one constant sample set per evaluation set. Samples leaving the moving image are padded, not
  discarded (discarding let the transform push bad samples out). L-BFGS on frozen samples overfits B-splines;
  `"sgd"`, with new samples every iteration, doesn't.
- **Pyramid:** isotropic reduced voxels; B-spline levels keep at least 32 voxels per axis (tiny levels diverged).
- **Bending:** our weight is Elastix's `TransformBendingEnergyPenalty` weight × 10⁻⁴. Elastix's default (1) is
  therefore effectively none.
  - Without bending, MI is exploited: the held-out MI of the result is better than that of the true deformation.
  - Synthetic sweep, MI: 1.68 mm at bending 0, 1.14 mm at 1, 1.08 mm at 3.
  - MSE is best without bending.
- **Elastix ASGD** (implemented faithfully, `"asgd"`): not better than `"sgd"` with our pyramid, and 2× slower.
  The best MI result so far (0.98 mm) combines Elastix's full B-spline configuration with bending 1, at 3× the time.
- **Rejected as defaults:**
  - automatic mask: MI 1.14 → 0.99 mm, but MSE 0.37 → 0.50 mm;
  - percentile MI limits: 1.14 vs 1.11 mm;
  - NGF `eta` 0.05–0.3: 1.27–1.36 vs 1.12 mm;
  - ×√2 grid schedule: MI 0.98 but MSE 0.67 mm.
- **Real data:** the Dixon is out-of-phase, so cancellation rims make objects look larger than in the T1. Intensity
  metrics tend to shrink them.
  - Rigid MI, Dixon → T1: (−0.68, 3.12, 5.86) mm; SimpleElastix: (−0.96, 2.81, 6.24) mm.
  - 2D B-spline with bending 1: no folding (determinant 0.93–1.05).
- **Sparse ROI interpolation**, leave-one-slice-out over 40 mm gaps, mean Dice:
  - copy of the nearest slice: 0.62;
  - shape-based: 0.71;
  - registration-based (2 levels × 32 iterations, 0.3 s per slice pair): 0.77.

  Running slice pairs in threads doesn't help (GIL).
- **Speed:**
  - about 50 ms per stochastic iteration;
  - warping a label map on 2.9 M voxels: 6 s (SimpleITK resampling through a displacement field, which keeps
    our clipped B-spline semantics; `to_sitk` extrapolates outside the grid);
  - folding check on 2.9 M voxels: 16 s.

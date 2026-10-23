"""transforms and transformations (transform with parameters)"""

from abc import ABC, abstractmethod

import numpy as np

from .image import Geometry, _tolist
from .splines import _spline_weights, bspline_coefficients


def _finite_jacobian(transformation, points=None, step=1e-2):
    """spatial Jacobian dT/dx of a transformation at points (npoint x ndim x ndim, world axes): finite
    differences along the grid axes, central inside the grid and one-sided at its borders (exact for global
    transforms, O(step^2) inside for B-splines)

    points: world points (default: the points of the transformation's grid)
    step: finite difference step (pixels of the transformation's grid)
    """
    geometry = transformation.geometry
    points = geometry.points if points is None else np.atleast_2d(np.asarray(points, dtype=float))
    coords = geometry.to_coords(points)
    ndim = points.shape[1]
    jac = np.empty((len(points), ndim, ndim))
    for k in range(ndim):
        upper = max(geometry.shape[k] - 1, 1)  # single-slice axes: forward differences
        low = np.where(coords[:, k] - step >= 0, coords[:, k] - step, coords[:, k])
        high = np.where(coords[:, k] + step <= upper, coords[:, k] + step, coords[:, k])
        c1, c2 = coords.copy(), coords.copy()
        c1[:, k], c2[:, k] = low, high
        delta = transformation(geometry.to_points(c2)) - transformation(geometry.to_points(c1))
        jac[:, :, k] = delta / (high - low)[:, np.newaxis]
    # d/dcoords -> d/dx
    return jac @ np.linalg.inv(geometry.matrix)


class Transformation:
    """Transform with parameters: maps points of the fixed image to the moving image"""

    def __init__(self, transform, geometry, parameters, center=None):
        """
        transform: Transform object, or type name or serialized transform (dict with a "type" key)
        geometry: output grid (fixed image)
        parameters: transform parameters
        center: rotation center (default: center of `geometry`)
        """
        if isinstance(transform, (str, dict)):
            transform = Transform.deserialize(transform)
        elif not isinstance(transform, Transform):
            raise TypeError(f"Not a Transform: {transform}")

        self.transform = transform
        self.geometry = geometry
        self.parameters = np.asarray(parameters)
        self._center = None if center is None else np.asarray(center, dtype=float)

    @property
    def center(self):
        return self.geometry.center if self._center is None else self._center

    @property
    def is_global(self):
        return self.transform.is_global

    def __repr__(self):
        clsname = type(self.transform).__name__
        if self.parameters.size > 12:
            return f"{clsname}({self.parameters.size} parameters, max |p| = {np.abs(self.parameters).max():.2})"
        values = ", ".join([f"{float(val):.2}" for val in self.parameters])
        return f"{clsname}([{values}])"

    def with_geometry(self, geometry):
        """same transformation, other output grid"""
        return Transformation(self.transform, geometry, self.parameters, center=self.center)

    def serialize(self):
        data = {
            "transform": self.transform.serialize(),
            "geometry": self.geometry.serialize(),
            "parameters": self.parameters.tolist(),
        }
        if self._center is not None:
            data["center"] = self._center.tolist()
        return data

    @classmethod
    def deserialize(cls, data):
        transform = Transform.deserialize(data["transform"])
        geometry = Geometry.deserialize(data["geometry"])
        return cls(transform, geometry, data["parameters"], center=data.get("center"))

    def __call__(self, points=None, **kwargs):
        """transform points (default: points of the output grid)"""
        points = self.geometry.points if points is None else points
        return self.transform(self.parameters, points, center=self.center, **kwargs)

    def jacobian(self, points=None):
        """spatial Jacobian dT/dx at points (npoint x ndim x ndim, world axes; default points: the grid's)"""
        points = self.geometry.points if points is None else np.atleast_2d(points)
        if self.is_global:
            return np.broadcast_to(self.affine()[0], (len(points),) + (points.shape[1],) * 2)
        if hasattr(self.transform, "spatial_jacobian"):
            return self.transform.spatial_jacobian(np.ravel(self.parameters), points, center=self.center)
        return _finite_jacobian(self, points)

    def affine(self):
        """matrix A and vector t of a global transformation: T(x) = A @ x + t"""
        if not self.is_global:
            raise ValueError(f"Not a global transform: {self.transform}")
        ndim = self.geometry.ndim
        points = self(np.vstack([np.zeros(ndim), np.eye(ndim)]))
        return (points[1:] - points[0]).T, points[0]

    def __array__(self, dtype=None, copy=None):
        """numpy converter"""
        return np.asarray(self.parameters, dtype=dtype)


class TransformationSequence:
    """composition of transformations: T(x) = T_n(...T_2(T_1(x)))"""

    def __init__(self, transformations):
        self.transformations = [as_transformation(trans) for trans in transformations]
        if not self.transformations:
            raise ValueError("Empty sequence of transformations")

    def __len__(self):
        return len(self.transformations)

    def __getitem__(self, index):
        return self.transformations[index]

    def __iter__(self):
        return iter(self.transformations)

    def __repr__(self):
        return f"TransformationSequence({self.transformations})"

    @property
    def geometry(self):
        return self.transformations[-1].geometry

    @property
    def parameters(self):
        return [trans.parameters for trans in self.transformations]

    @property
    def is_global(self):
        return all(trans.is_global for trans in self.transformations)

    def with_geometry(self, geometry):
        return TransformationSequence([trans.with_geometry(geometry) for trans in self.transformations])

    def __call__(self, points=None):
        points = self.geometry.points if points is None else points
        for trans in self.transformations:
            points = trans(points)
        return points

    def jacobian(self, points=None):
        """spatial Jacobian (chain rule): J_n(T_{n-1}(x)) ... J_1(x)"""
        points = self.geometry.points if points is None else np.atleast_2d(points)
        jac = None
        for k, trans in enumerate(self.transformations):
            current = trans.jacobian(points)
            jac = current if jac is None else current @ jac
            if k < len(self.transformations) - 1:
                points = trans(points)
        return jac

    def affine(self):
        A, t = np.eye(self.geometry.ndim), np.zeros(self.geometry.ndim)
        for trans in self.transformations:
            A_, t_ = trans.affine()
            A, t = A_ @ A, A_ @ t + t_
        return A, t

    def serialize(self):
        return [trans.serialize() for trans in self.transformations]

    @classmethod
    def deserialize(cls, data):
        return cls([Transformation.deserialize(item) for item in data])


def as_transformation(obj):
    """Transformation or TransformationSequence from an object or its serialized form"""
    if isinstance(obj, (Transformation, TransformationSequence)):
        return obj
    elif isinstance(obj, dict):
        return Transformation.deserialize(obj)
    elif isinstance(obj, (list, tuple)):
        sequence = TransformationSequence.deserialize(obj)
        return sequence[0] if len(sequence) == 1 else sequence
    raise TypeError(f"Not a transformation: {obj}")


class Transform(ABC):
    """Base class for geometric transform"""

    # affine transform (with fixed matrix and translation)
    is_global = False

    @abstractmethod
    def transform(self, parameters, points, **kwargs):
        pass

    def settings(self):
        """initialization arguments (serializable)"""
        return {}

    def serialize(self):
        from .config import component_name  # registry (imports the transforms)

        return {"type": component_name("transform", self), **self.settings()}

    @classmethod
    def deserialize(cls, data):
        if isinstance(data, str):
            data = {"type": data}
        from .config import get_component  # registry (imports the transforms)

        data = dict(data)
        transform = get_component("transform", data.pop("type"))
        return transform(**data)

    def parse(self, parameters):
        raise NotImplementedError("'parse' is not implemented for this transform")

    def pack(self, *args, **kwargs):
        raise NotImplementedError("'pack' is not implemented for this transform")

    def __call__(self, parameters, points, **kwargs):
        parameters = np.ravel(parameters)
        points = np.atleast_2d(points)
        return self.transform(parameters, points, **kwargs)

    def init(self, ndim):
        raise NotImplementedError("'init' is not implemented for this transform")

    def setup(self, geometry, level=0):
        """prepare the transform for a fixed image geometry (e.g. a control grid)

        level: pyramid level, relative to the finest level used (0)
        """

    def refine(self, previous, parameters):
        """parameters of this transform equivalent to `parameters` of the transform of the previous level"""
        return parameters

    def with_translation(self, parameters, offset):
        """parameters with the given translation (None if not supported)"""
        if getattr(self, "axis", None) is not None:
            return None
        parameters = np.array(parameters, dtype=float)
        parameters[-len(offset):] = offset
        return parameters

    def scales(self, parameters, radius):
        """parameter scales: one unit of each scaled parameter moves the points by about 1 mm

        radius: typical distance of the points to the center (mm)
        """
        return np.ones(len(parameters))

    def __str__(self):
        return str(type(self).__name__)


class TransformTranslation(Transform):
    """Translation transform"""

    def __init__(self, axis=None, init=None):
        self.axis = axis
        self._init = init

    def parse(self, parameters):
        return {"b": parameters}

    def pack(self, b):
        return b

    is_global = True

    def settings(self):
        return {"axis": _tolist(self.axis), "init": _tolist(self._init)}

    def init(self, ndim):
        if self._init is not None:
            return self._init
        elif self.axis is not None:
            return np.zeros(ndim - 1)
        return np.zeros(ndim)

    def inverse(self, parameters):
        return -parameters

    def transform(self, parameters, points, jac=False, **kwargs):
        if not jac:
            return translation(parameters, points, axis=self.axis)
        return translation(parameters, points, jac=True, axis=self.axis)


class TransformRigid(Transform):
    """Rigid transform

    parameters:
        2d: (phi, bx, by)
        3d: Euler angles and translation (ax, ay, az, bx, by, bz), with R = Rz @ Rx @ Ry (as ITK's Euler3DTransform)
        3d with fixed axis: (phi, b1, b2), rotation about `axis` and translation normal to it
    """

    def __init__(self, axis=None, center=None, init=None):
        self.center = None if center is None else np.asarray(center)
        if np.isscalar(axis) and axis in [0, 1, 2]:
            axis = np.eye(3)[axis]
        elif np.isscalar(axis):
            raise ValueError(axis)
        self.axis = None if axis is None else np.asarray(axis) / np.linalg.norm(axis)
        self._init = None if init is None else np.asarray(init)

    is_global = True

    def settings(self):
        return {"axis": _tolist(self.axis), "center": _tolist(self.center), "init": _tolist(self._init)}

    def init(self, ndim):
        if self._init is not None:
            return self._init
        elif ndim == 2:
            return self.pack(phi=0, b=[0, 0])
        elif ndim == 3 and self.axis is not None:
            return self.pack(phi=0, b=[0, 0])
        elif ndim == 3:
            return self.pack(angles=[0, 0, 0], b=[0, 0, 0])
        raise NotImplementedError(f"Invalid number of dimensions: {ndim}")

    def pack(self, **kwargs):
        ndim = len(kwargs["b"])
        if "angles" in kwargs:
            return np.r_[kwargs["angles"], kwargs["b"]]
        elif ndim == 2:
            return np.r_[kwargs["phi"], kwargs["b"]]
        elif ndim == 3 and self.axis is not None:
            return np.r_[kwargs["phi"], kwargs["b"]]
        raise NotImplementedError(f"Invalid number of dimensions: {ndim}")

    def parse(self, parameters):
        if len(parameters) == 3:
            return {"phi": parameters[0], "b": parameters[1:]}
        elif len(parameters) == 6:
            return {"angles": parameters[:3], "b": parameters[3:]}
        raise NotImplementedError(f"Invalid parameters: {parameters}")

    def transform(self, parameters, points, center=None, jac=False, **kwargs):
        if self.center is not None:
            # use pre defined rotation center
            center = self.center
        elif center is None:
            center = 0

        ndim = points.shape[1]
        if ndim == 2:
            if not jac:
                new = rigidtransform2d(parameters, points - center, jac=False)
                return new + center
            else:
                new, jmat = rigidtransform2d(parameters, points - center, jac=True)
                return new + center, jmat

        elif ndim == 3:
            if not jac:
                new = rigidtransform3d(
                    parameters, points - center, axis=self.axis, jac=False
                )
                return new + center
            else:
                new, jmat = rigidtransform3d(
                    parameters, points - center, axis=self.axis, jac=True
                )
                return new + center, jmat
        else:
            raise NotImplementedError(f"Invalid number of dimensions: {ndim}")

    def scales(self, parameters, radius):
        """rotation angles scaled by the radius, translations unchanged"""
        nrot = 3 if len(parameters) == 6 else 1
        return np.r_[np.full(nrot, radius), np.ones(len(parameters) - nrot)]

    def inverse(self, parameters):
        """parameters of the inverse transform (same center)"""
        if len(parameters) == 3:
            # 2d, or 3d with fixed axis (same formula in the plane normal to the axis)
            return rigidinverse2d(parameters)
        return rigidinverse3d(parameters)


class TransformAffine(Transform):
    """Affine transform"""

    def __init__(self, axis=None, init=None):
        self.axis = axis
        self._init = init

    is_global = True

    def settings(self):
        return {"axis": _tolist(self.axis), "init": _tolist(self._init)}

    def init(self, ndim):
        if self._init is not None:
            return self._init
        elif not ndim in [2, 3]:
            raise NotImplementedError(ndim)
        elif self.axis is not None:
            ndim = 2
        A = np.eye(ndim)
        b = np.zeros(ndim)
        return self.pack(A, b)

    def scales(self, parameters, radius):
        """matrix coefficients scaled by the radius, translations unchanged"""
        nmat = len(self.parse(np.asarray(parameters))["A"].flat)
        return np.r_[np.full(nmat, radius), np.ones(len(parameters) - nmat)]

    def parse(self, parameters):
        ndim = {2: 1, 6: 2, 12: 3}.get(len(parameters), None)
        if ndim is None:
            # compute number of dimensions
            ndim = (int(np.sqrt(4 * len(parameters) + 1)) - 1) // 2
        return {
            "A": parameters[: ndim**2].reshape(ndim, ndim),
            "b": parameters[ndim**2 :],
        }

    def pack(self, A, b):
        return np.r_[A.flat, b]

    def transform(self, parameters, points, jac=False, center=None, **kwargs):
        if center is None:
            center = 0
        if not jac:
            new = affinetransform2d3d(
                parameters, points - center, jac=False, axis=self.axis
            )
            return new + center
        # else
        new, jmat = affinetransform2d3d(
            parameters, points - center, jac=True, axis=self.axis
        )
        return new + center, jmat

    def inverse(self, parameters):
        parsed = self.parse(parameters)
        invA = np.linalg.inv(parsed["A"])
        invb = invA @ parsed["b"]
        return np.r_[invA.ravel(), -invb]


class TransformBSpline(Transform):
    """Free-form deformation: cubic B-spline displacement, T(x) = x + u(x)

    u(x) = sum_k beta3((x - x_k) / h) d_k, on a control grid aligned with the fixed image axes, covering its
    domain (with one control point before and two after along each axis, as ITK's BSplineTransform)

    grid_spacing: control point spacing (mm), scalar or per axis; adjusted so that the grid fits the domain
    schedule: grid spacing factor at pyramid level k: schedule ** k (as Elastix's default GridSpacingSchedule);
        the displacement of each level initializes the (finer) grid of the next level; 1: same grid at all levels
    bending: weight of the bending energy penalty (smoothness regularization, default 0): the penalty is
        bending x mean squared second derivatives of the displacement x (100 mm)^2, independent of the grid
        spacing. It scales as amplitude^2 / wavelength^4 of the deformation: e.g. 2 mm over 100 mm gives ~0.3,
        2 mm over 30 mm ~ 30. With MI or NGF (costs of order 1), 0.1-1 regularizes smooth deformations
        (and prevents the exploitation of the MI estimate); MSE costs depend on the intensities.
    axis: axis of the fixed image along which there is no displacement (e.g. 2: in-plane deformation only, slices
        stay slices, as needed by ROIs drawn on some slices); None: displacement along all axes
    grid: control grid (dict: origin, spacing, shape, direction), set from the fixed image by `setup`
    init: initial parameters (displacements of the control points, mm)

    parameters: displacement components along the fixed image axes (number of components x number of control
        points), in mm
    """

    is_global = False

    # reference length of the bending penalty (mm): weights of order 1 for costs of order 1
    BENDING_LENGTH = 100.0

    def __init__(self, grid_spacing=20.0, schedule=2.0, bending=0.0, axis=None, grid=None, init=None):
        self.axis = None if axis is None else int(axis)
        self.grid_spacing = grid_spacing
        self.schedule = float(schedule)
        self.bending = float(bending)
        self.grid = None if grid is None else {key: np.asarray(value, dtype=float) for key, value in grid.items()}
        if self.grid is not None:
            self.grid["shape"] = self.grid["shape"].astype(int)
        self._init = None if init is None else np.asarray(init, dtype=float)

    def settings(self):
        grid = None if self.grid is None else {key: _tolist(value) for key, value in self.grid.items()}
        return {
            "grid_spacing": _tolist(self.grid_spacing), "schedule": self.schedule, "bending": self.bending,
            "axis": self.axis, "grid": grid, "init": _tolist(self._init),
        }

    def setup(self, geometry, level=0):
        if self.grid is not None:
            return
        ndim = geometry.ndim
        extent = (geometry.shape - 1) * geometry.spacing  # first to last pixel centers
        spacing = np.broadcast_to(np.asarray(self.grid_spacing, dtype=float), (ndim,))
        if self._init is None:
            spacing = spacing * self.schedule**level
        mesh = np.maximum(np.ceil(extent / spacing - 1e-6), 1).astype(int)
        spacing = np.where(extent > 0, extent / mesh, 1.0)
        origin = geometry.origin - geometry.direction @ spacing  # one control point before
        self.grid = {"origin": origin, "spacing": spacing, "shape": mesh + 3, "direction": geometry.direction}

    @property
    def ncontrol(self):
        return int(np.prod(self.grid["shape"]))

    @property
    def components(self):
        """fixed image axes along which there is a displacement"""
        return [a for a in range(len(self.grid["shape"])) if a != self.axis]

    @property
    def basis(self):
        """world directions of the displacement components (ndim x number of components)"""
        return self.grid["direction"][:, self.components]

    def control_points(self):
        """world coordinates of the control points"""
        shape = self.grid["shape"]
        indices = np.reshape(np.indices(shape), (len(shape), -1)).T
        return self.grid["origin"] + (indices * self.grid["spacing"]) @ self.grid["direction"].T

    def refine(self, previous, parameters):
        """coefficients interpolating the displacement of the previous (coarser) grid at the control points"""
        if previous.grid is not None and all(
            np.allclose(previous.grid[key], self.grid[key]) for key in ("origin", "spacing", "shape")
        ):
            return parameters
        shape = tuple(self.grid["shape"])
        disp = previous.displacement(parameters, self.control_points()) @ self.basis
        coeffs = [bspline_coefficients(disp[:, a].reshape(shape), 3) for a in range(disp.shape[1])]
        return np.concatenate([c.ravel() for c in coeffs])

    def init(self, ndim):
        if self.grid is None:
            raise ValueError("TransformBSpline: no control grid (call `setup` with the fixed image geometry)")
        if self._init is not None:
            return self._init
        return np.zeros(len(self.components) * self.ncontrol)

    def grid_coordinates(self, points):
        """control grid coordinates of points (clipped to the domain [1, shape - 2): the 4 taps of each point stay
        in the grid)"""
        coords = (points - self.grid["origin"]) @ self.grid["direction"] / self.grid["spacing"]
        return np.clip(coords, 1, np.nextafter(self.grid["shape"] - 2.0, 0))

    def weights(self, points):
        """flat control point indices and weights of the points (npoint x 4^ndim), cached for the last
        points array (they do not depend on the parameters)"""
        cached = getattr(self, "_weights", (None, None))
        if cached[0] is points:
            return cached[1]
        result = self._weights_of(points)
        self._weights = (points, result)
        return result

    def _weights_of(self, points):
        coords = self.grid_coordinates(points)
        npoint, ndim = coords.shape
        shape = self.grid["shape"]
        strides = [int(np.prod(shape[d + 1:])) for d in range(ndim)]
        flat = np.zeros((npoint,) + (1,) * ndim, dtype=np.intp)
        weights = np.ones((npoint,) + (1,) * ndim)
        for d in range(ndim):
            first, w, _ = _spline_weights(coords[:, d], 3, shape[d])
            index = first[:, np.newaxis].astype(np.intp) + np.arange(4)
            newshape = [npoint] + [1] * ndim
            newshape[d + 1] = 4
            flat = flat + (index * strides[d]).reshape(newshape)
            weights = weights * w.reshape(newshape)
        return flat.reshape(npoint, -1), weights.reshape(npoint, -1)

    def _displacement(self, coeffs, flat, weights):
        """world displacement from the component coefficients"""
        local = np.stack([(coeffs[a][flat] * weights).sum(axis=1) for a in range(len(coeffs))], axis=1)
        return local @ self.basis.T

    def spatial_jacobian(self, parameters, points, chunk=2**15, **kwargs):
        """spatial Jacobian dT/dx at points (npoint x ndim x ndim, world axes), from the derivatives of the cubic
        B-spline weights (zero across the borders of the control grid domain, where coordinates are clipped)"""
        ndim = points.shape[1]
        coeffs = np.reshape(parameters, (len(self.components), -1))
        shape = self.grid["shape"]
        # world derivative of the grid coordinates: dc_d / dx = direction[:, d] / spacing[d]
        dcoords = (self.grid["direction"] / self.grid["spacing"]).T  # ndim (d) x ndim (x)
        jac = np.empty((len(points), ndim, ndim))
        for start in range(0, len(points), chunk):
            index = slice(start, start + chunk)
            raw = (points[index] - self.grid["origin"]) @ self.grid["direction"] / self.grid["spacing"]
            npoint = len(raw)
            flat, taps = self._taps(points[index])
            # du_local / dc_d: tap values contracted axis by axis (last first) with the weights, or their
            # derivative along d (separable)
            dlocal = np.empty((npoint, len(coeffs), ndim))
            for a, c in enumerate(coeffs):
                values = c[flat].reshape((npoint,) + (4,) * ndim)
                for d in range(ndim):
                    out = values
                    for e in range(ndim - 1, -1, -1):
                        _, w, dw = taps[e]
                        out = np.einsum("n...k,nk->n...", out, dw if e == d else w)
                    dlocal[:, a, d] = out
            inside = (raw >= 1 - 1e-9) & (raw <= shape - 2 + 1e-9)  # not clipped
            dlocal *= inside[:, np.newaxis, :]
            # world: J = I + basis @ dlocal @ dcoords
            jac[index] = np.eye(ndim) + np.einsum("ia,nad,dj->nij", self.basis, dlocal, dcoords)
        return jac

    def _taps(self, points):
        """flat indices of the 4^ndim taps of the points (npoint x 4^ndim) and the per axis taps
        (first index, weights, derivative weights)"""
        coords = self.grid_coordinates(points)
        npoint, ndim = coords.shape
        shape = self.grid["shape"]
        strides = [int(np.prod(shape[d + 1:])) for d in range(ndim)]
        taps = [_spline_weights(coords[:, d], 3, shape[d]) for d in range(ndim)]
        flat = np.zeros((npoint,) + (1,) * ndim, dtype=np.intp)
        for d, (first, _, _) in enumerate(taps):
            newshape = [npoint] + [1] * ndim
            newshape[d + 1] = 4
            flat = flat + ((first[:, np.newaxis].astype(np.intp) + np.arange(4)) * strides[d]).reshape(newshape)
        return flat.reshape(npoint, -1), taps

    def displacement(self, parameters, points, chunk=2**15):
        """displacement u(x) at points"""
        coeffs = np.reshape(parameters, (len(self.components), -1))
        if len(points) <= chunk:
            return self._displacement(coeffs, *self.weights(points))
        # large point sets (not cached): separable evaluation, axis by axis
        ndim = points.shape[1]
        local = np.empty((len(points), len(coeffs)))
        for start in range(0, len(points), chunk):
            index = slice(start, start + chunk)
            flat, taps = self._taps(points[index])
            for a, c in enumerate(coeffs):
                out = c[flat].reshape((len(flat),) + (4,) * ndim)
                for e in range(ndim - 1, -1, -1):
                    out = np.einsum("n...k,nk->n...", out, taps[e][1])
                local[index, a] = out
        return local @ self.basis.T

    def transform(self, parameters, points, jac=False, **kwargs):
        if not jac:
            return points + self.displacement(parameters, points)
        flat, weights = self.weights(points)
        coeffs = np.reshape(parameters, (len(self.components), -1))
        disp = self._displacement(coeffs, flat, weights)
        return points + disp, SparseJacobian(flat, weights, self.ncontrol, self.basis)

    def penalty(self, parameters, gradient=False):
        """bending energy: weight x mean squared second derivatives of the displacement at the control points"""
        if not self.bending:
            return (0.0, np.zeros(len(parameters))) if gradient else 0.0
        shape = tuple(self.grid["shape"])
        ndim = len(shape)
        ncomp = len(self.components)
        coeffs = np.reshape(parameters, (ncomp,) + shape)
        spacing = self.grid["spacing"]
        norm = self.bending * self.BENDING_LENGTH**2 / self.ncontrol
        energy = 0.0
        grad = np.zeros_like(coeffs)
        for i in range(ndim):
            for j in range(i, ndim):
                factor = norm * (1 if i == j else 2) / (spacing[i] * spacing[j]) ** 2
                for a in range(ncomp):
                    diff = _second_difference(coeffs[a], i, j)
                    if diff is None:
                        continue
                    energy += factor * np.sum(diff**2)
                    if gradient:
                        grad[a] += _second_difference_adjoint(2 * factor * diff, i, j, shape)
        return (energy, grad.ravel()) if gradient else energy


def _second_difference(array, i, j):
    """second (i == j) or mixed (i != j) central difference along axes i, j (interior), None if too small"""
    if array.shape[i] < 3 or array.shape[j] < 3:
        return None
    ndim = array.ndim
    if i == j:
        def sl(offset):
            index = [slice(None)] * ndim
            index[i] = slice(1 + offset, array.shape[i] - 1 + offset)
            return tuple(index)
        return array[sl(-1)] - 2 * array[sl(0)] + array[sl(1)]

    def sl(oi, oj):
        index = [slice(None)] * ndim
        index[i] = slice(1 + oi, array.shape[i] - 1 + oi)
        index[j] = slice(1 + oj, array.shape[j] - 1 + oj)
        return tuple(index)
    return (array[sl(1, 1)] - array[sl(1, -1)] - array[sl(-1, 1)] + array[sl(-1, -1)]) / 4


def _second_difference_adjoint(residual, i, j, shape):
    """adjoint of `_second_difference`"""
    out = np.zeros(shape)
    ndim = len(shape)
    if i == j:
        for offset, factor in [(-1, 1), (0, -2), (1, 1)]:
            index = [slice(None)] * ndim
            index[i] = slice(1 + offset, shape[i] - 1 + offset)
            out[tuple(index)] += factor * residual
        return out
    for oi, oj, factor in [(1, 1, 0.25), (1, -1, -0.25), (-1, 1, -0.25), (-1, -1, 0.25)]:
        index = [slice(None)] * ndim
        index[i] = slice(1 + oi, shape[i] - 1 + oi)
        index[j] = slice(1 + oj, shape[j] - 1 + oj)
        out[tuple(index)] += factor * residual
    return out


class SparseJacobian:
    """jacobian of the transformed points w/r to B-spline coefficients: the displacement of point n along the
    world direction basis[:, a] depends on the coefficients (a, flat[n, t]) with weights[n, t]"""

    def __init__(self, flat, weights, ncontrol, basis):
        self.flat, self.weights, self.ncontrol, self.basis = flat, weights, ncontrol, np.asarray(basis)

    def rmatvec(self, gpoints, valid=None):
        """J^T g: gpoints (nvalid x ndim) gradient w/r to the (valid) points -> gradient w/r to the parameters"""
        flat, weights = (self.flat, self.weights) if valid is None else (self.flat[valid], self.weights[valid])
        glocal = gpoints @ self.basis
        grad = [
            np.bincount(flat.ravel(), weights=(weights * glocal[:, a, np.newaxis]).ravel(), minlength=self.ncontrol)
            for a in range(glocal.shape[1])
        ]
        return np.concatenate(grad)


def normalaxes3d(n):
    """return axes normal to n"""
    if np.isscalar(n) and n in [0, 1, 2]:
        n = np.eye(3)[n]
    i = np.argmax(np.abs(n))
    j = (i + 1) % 3
    n2 = np.zeros(3)
    n2[j] = n[i]
    n2[i] = -n[j]
    n3 = np.cross(n, n2)
    return n2 / np.linalg.norm(n2), n3 / np.linalg.norm(n3)


def translation(parameters, points, axis=None, jac=False):
    """tranlation, with option fixed axis"""
    ndim = points.shape[1]
    if ndim == 3 and axis is not None:
        assert len(parameters) == 2
        nx = np.c_[normalaxes3d(axis)]
        b = nx @ parameters
    else:
        nx = np.eye(ndim)
        b = parameters

    new = points + b
    if not jac:
        return new

    # else, also return jacobian (npoint x ndim x nparam)
    npoint = len(points)
    jmat = np.stack([np.broadcast_to(n, (npoint, ndim)) for n in nx.T], axis=-1)
    return new, jmat


def rigidtransform2d(parameters, points, jac=False):
    """2d transform"""
    phi = parameters[0]
    b = parameters[1:]

    A = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    new = points @ A.T + b

    if not jac:
        return new

    # else, compute jacobian
    dA = np.array([[-np.sin(phi), -np.cos(phi)], [np.cos(phi), -np.sin(phi)]])
    jacA = points @ dA.T
    npoint = len(points)
    jacb1 = np.broadcast_to(np.array([[1, 0]]), (npoint, 2))
    jacb2 = np.broadcast_to(np.array([[0, 1]]), (npoint, 2))
    jac = np.stack([jacA, jacb1, jacb2], axis=2)
    return new, jac


def rigidinverse2d(parameters):
    """Return parameters of inverse transform"""
    phi, b = parameters[0], parameters[1:]
    A_ = np.array([[np.cos(-phi), -np.sin(-phi)], [np.sin(-phi), np.cos(-phi)]])
    b_ = -A_ @ b
    return [-phi, b_[0], b_[1]]


def euler_matrices(angles):
    """rotation matrices about x, y, z"""
    ca, cb, cg = np.cos(angles)
    sa, sb, sg = np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    dRx = np.array([[0, 0, 0], [0, -sa, -ca], [0, ca, -sa]])
    dRy = np.array([[-sb, 0, cb], [0, 0, 0], [-cb, 0, -sb]])
    dRz = np.array([[-sg, -cg, 0], [cg, -sg, 0], [0, 0, 0]])
    return (Rx, Ry, Rz), (dRx, dRy, dRz)


def euler_to_matrix(angles):
    """rotation matrix R = Rz @ Rx @ Ry (ITK's Euler3DTransform)"""
    (Rx, Ry, Rz), _ = euler_matrices(angles)
    return Rz @ Rx @ Ry


def matrix_to_euler(R):
    """Euler angles (ax, ay, az) of a rotation matrix R = Rz @ Rx @ Ry"""
    ax = np.arcsin(np.clip(R[2, 1], -1, 1))
    ay = np.arctan2(-R[2, 0], R[2, 2])
    az = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([ax, ay, az])


def rigidtransform3d(parameters, points, axis=None, jac=False):
    """3d Rigid transform

    parameters:
        axis is None: Euler angles (ax, ay, az) and translation (bx, by, bz), R = Rz @ Rx @ Ry
        axis is set: angle (phi) about `axis`, translation (b1, b2) in the plane normal to `axis`
    """
    parameters = np.asarray(parameters, dtype=float)
    if axis is None:
        assert len(parameters) == 6
        return eulertransform3d(parameters, points, jac=jac)

    assert len(parameters) == 3
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    nx = np.c_[normalaxes3d(n)]
    phi = parameters[0]
    b = nx @ parameters[1:]

    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    dotnX = np.dot(points, n)
    projnX = np.c_[dotnX] * n
    crossnX = np.cross(n, points)
    new = cosphi * points + (1 - cosphi) * projnX + sinphi * crossnX + b

    if not jac:
        return new

    # else, compute jacobian
    jacphi = -sinphi * points + sinphi * projnX + cosphi * crossnX
    npoint = len(points)
    jacb = [np.broadcast_to(vec, (npoint, 3)) for vec in nx.T]
    return new, np.stack([jacphi] + jacb, axis=2)


def eulertransform3d(parameters, points, jac=False):
    """3d rigid transform with Euler angles: R = Rz @ Rx @ Ry, parameters (ax, ay, az, bx, by, bz)"""
    (Rx, Ry, Rz), (dRx, dRy, dRz) = euler_matrices(parameters[:3])
    b = parameters[3:]
    new = points @ (Rz @ Rx @ Ry).T + b
    if not jac:
        return new
    dR = [Rz @ dRx @ Ry, Rz @ Rx @ dRy, dRz @ Rx @ Ry]
    npoint = len(points)
    jacR = [points @ mat.T for mat in dR]
    jacb = [np.broadcast_to(vec, (npoint, 3)) for vec in np.eye(3)]
    return new, np.stack(jacR + jacb, axis=2)


def rigidinverse3d(parameters):
    """Return parameters of inverse 3d rigid transform (Euler angles)"""
    parameters = np.asarray(parameters, dtype=float)
    R = euler_to_matrix(parameters[:3])
    return np.r_[matrix_to_euler(R.T), -R.T @ parameters[3:]]


def affinetransform2d3d(parameters, points, axis=None, jac=False):
    """affine transform 2d/3d"""
    ndim = points.shape[1]
    if not ndim in [2, 3]:
        raise ValueError(f"Only 2d or 3d, not {ndim}d")
    elif ndim == 2:
        assert len(parameters) == 6
        A = np.reshape(parameters[:4], (2, 2))
        b = parameters[4:]
        nx = np.eye(2)
    elif axis is not None:
        assert len(parameters) == 6
        if np.isscalar(axis) and axis in [0, 1, 2]:
            axis = np.eye(3)[axis]
        nx = np.c_[normalaxes3d(axis)]
        _A = np.reshape(parameters[:4], (2, 2))
        _b = parameters[4:]
        A = nx @ _A @ nx.T + np.outer(axis, axis)
        b = nx @ _b
    else:
        assert len(parameters) == 12
        A = np.reshape(parameters[:9], (3, 3))
        b = parameters[9:]
        nx = np.eye(3)

    # transform points
    new = points @ A.T + b

    if not jac:
        return new

    npoint = len(points)
    jacA = [points @ np.outer(n1, n2).T for n1 in nx.T for n2 in nx.T]
    jacb = [np.broadcast_to(n, (npoint, ndim)) for n in nx.T]
    jac = np.stack(jacA + jacb, axis=2)
    return new, jac

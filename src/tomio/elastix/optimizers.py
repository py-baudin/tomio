"""optimizers"""

import logging
from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np

LOGGER = logging.getLogger("tomio.elastix")


class Optimizer(ABC):
    """Base class for optimizers"""

    @abstractmethod
    def optimize(self, objective, init, **kwargs):
        pass

    def __call__(self, objective, init, **kwargs):
        return self.optimize(objective, init, **kwargs)

    def __str__(self):
        return str(type(self).__name__)


class OptimizerLBFGS(Optimizer):
    """Limited-memory BFGS with backtracking line search (samples fixed during the optimization)

    The parameters are scaled (one unit moves the points by about 1 mm).

    maxiter: maximum number of iterations
    memory: number of corrections kept
    gtol: stop when the largest gradient component is below gtol
    ftol: stop when the relative cost decrease is below ftol
    xtol: stop when the largest parameter change is below xtol (~ mm)
    maxstep: largest parameter change of the first iteration (~ mm)
    """

    def __init__(self, maxiter=500, memory=10, gtol=1e-8, ftol=1e-7, xtol=1e-3, maxstep=1.0):
        self.opts = {"maxiter": maxiter, "memory": memory, "gtol": gtol, "ftol": ftol, "xtol": xtol, "maxstep": maxstep}

    def optimize(self, objective, init):
        sol, info = lbfgs(lambda x: objective(x, gradient=True), init, **self.opts)
        return sol, OptimResults(**info)


class OptimizerSGD(Optimizer):
    """Stochastic gradient descent with new samples at each iteration

    Iteration k moves the (scaled) parameters by -gain_k * gradient / max(|gradient|, g0), with
    gain_k = a / (k + 1 + A)^alpha, |.| the largest component and g0 that of the first gradient: no step is larger
    than gain_k (~ mm), and the steps shrink when the gradient becomes smaller than at the start. New samples at
    each iteration avoid fitting the transform to a fixed set of samples (needed for transforms with many
    parameters, e.g. B-splines); fixed number of iterations. See "asgd" for Elastix's optimizer.

    niter: number of iterations (per pyramid level)
    a, A, alpha: gain sequence (A: stability constant, alpha: decay exponent; 0.602 as in Elastix)
    """

    def __init__(self, niter=256, a=4.0, A=20, alpha=0.602):
        self.niter, self.a, self.A, self.alpha = niter, a, A, alpha

    def optimize(self, objective, init):
        x = np.asarray(init, dtype=float).copy()
        cost, norm = np.nan, None
        for k in range(self.niter):
            if k:
                objective.resample()
            cost, grad = objective(x, gradient=True)
            current = np.max(np.abs(grad))
            if norm is None:
                norm = current
            if not np.isfinite(cost) or max(current, norm) == 0:
                break
            x -= self.a / (k + 1 + self.A) ** self.alpha * grad / max(current, norm)
        return x, OptimResults(fun=cost, nit=k + 1, nfev=k + 1, message="Number of iterations reached", success=True)


class OptimizerASGD(Optimizer):
    """Adaptive stochastic gradient descent, as Elastix's AdaptiveStochasticGradientDescent with automatic
    parameter estimation (method "Original"; Klein et al., IJCV 2009)

    Iteration k: x -= a / (t_k + A + 1)^alpha * g_k (raw gradients of the cost, new samples at each iteration);
    adaptive time t_{k+1} = max(0, t_k + f(-g_k . g_{k-1})), f a sigmoid from fmin to fmax with f(0) = 0.
    At the start of each pyramid level, a, alpha (= 1), fmin, fmax (= 1) and the sigmoid scale omega are
    estimated from the transform Jacobian at `njacobian` grid points and from `ngradient` pairs of gradients
    ("exact": `nexact` grid samples; stochastic: the sampler) at random perturbations of the parameters, so that
    the first steps move the points by about `maxstep`.

    niter: number of iterations (per pyramid level; Elastix's default map: 256)
    A: stability constant (Elastix: 20)
    maxstep: largest expected displacement of a step (mm; Elastix's MaximumStepLength, default: mean in-plane
        spacing of the full resolution fixed and moving images)
    njacobian: number of Jacobian measurements (default: max(1000, number of parameters), as Elastix)
    nexact: number of samples of the "exact" gradients (Elastix: 100000)
    ngradient: number of gradient measurements (default: automatic, at least 2, as Elastix)
    sigmoid_scale_factor: Elastix's SigmoidScaleFactor (0.1)
    adaptive: adaptive time (Elastix's UseAdaptiveStepSizes, default true); False: t_k += (fmax + fmin) / 2
    seed: seed of the parameter perturbations
    """

    def __init__(self, niter=256, A=20.0, maxstep=None, njacobian=None, nexact=100000, ngradient=None,
                 sigmoid_scale_factor=0.1, adaptive=True, seed=0):
        self.niter, self.A, self.maxstep = niter, float(A), maxstep
        self.njacobian, self.nexact, self.ngradient = njacobian, nexact, ngradient
        self.sigmoid_scale_factor, self.adaptive, self.seed = sigmoid_scale_factor, bool(adaptive), seed
        self.default_maxstep = 1.0  # set by the pipeline: mean in-plane spacing of the images
        self.estimates = []  # estimated settings, per pyramid level

    def estimate(self, objective, x):
        """Elastix's AutomaticParameterEstimationOriginal: a, alpha, fmin, fmax, omega"""
        delta = self.maxstep if self.maxstep is not None else self.default_maxstep
        njacobian = self.njacobian if self.njacobian is not None else max(1000, len(x))
        TrC, TrCC, maxJJ, maxJCJ = jacobian_terms(objective.jacobian_samples(x, njacobian), len(x))
        ngradient = self.ngradient
        if ngradient is None:
            K = 1.5
            ngradient = int(np.ceil(8.0 * TrCC / TrC / TrC / (K - 1) / (K - 1))) if TrCC > 1e-14 and TrC > 1e-14 else 2
            ngradient = max(2, ngradient)
        sigma4 = delta / np.sqrt(maxJJ) if maxJJ > 1e-14 else 0.0
        rng = np.random.default_rng(self.seed)
        gg = ee = 0.0
        for _ in range(ngradient):
            perturbed = x + sigma4 * rng.normal(size=len(x))
            exact = objective.exact_gradient(perturbed, self.nexact)
            objective.resample()
            _, approx = objective(perturbed, gradient=True)
            gg += exact @ exact
            ee += (exact - approx) @ (exact - approx)
        gg, ee = gg / ngradient, ee / ngradient
        sigma1 = np.sqrt(gg / TrC) if gg > 1e-14 and TrC > 1e-14 else 0.0
        sigma3 = np.sqrt(ee / TrC) if ee > 1e-14 and TrC > 1e-14 else 0.0
        a_max = self.A * delta / sigma1 / np.sqrt(maxJCJ) if sigma1 > 1e-14 and maxJCJ > 1e-14 else 0.0
        noisefactor = sigma1**2 / (sigma1**2 + sigma3**2 + 1e-14)
        estimate = {
            "a": a_max * noisefactor, "alpha": 1.0, "fmax": 1.0, "fmin": -0.99 + 0.98 * noisefactor,
            "omega": max(1e-14, self.sigmoid_scale_factor * sigma3**2 * np.sqrt(TrCC)),
            "maxstep": delta, "ngradient": ngradient, "njacobian": njacobian,
        }
        self.estimates.append(estimate)
        return estimate

    @staticmethod
    def sigmoid(x, fmin, fmax, omega):
        """ITK's sigmoid with beta such that f(0) = 0"""
        beta = omega * np.log(-fmax / fmin)
        z = np.clip(-(x - beta) / omega, -700, 700)
        return (fmax - fmin) / (1 + np.exp(z)) + fmin

    def optimize(self, objective, init):
        x = np.asarray(init, dtype=float).copy()
        cost, _ = objective(x, gradient=True)  # first evaluation at the start (defines the sample overlap)
        est = self.estimate(objective, x)
        time, previous = 0.0, None
        for k in range(self.niter):
            objective.resample()
            cost, grad = objective(x, gradient=True)
            if not np.isfinite(cost):
                break
            x -= est["a"] / (time + self.A + 1.0) ** est["alpha"] * grad
            if self.adaptive:
                if previous is not None:
                    time = max(0.0, time + float(self.sigmoid(-(previous @ grad), est["fmin"], est["fmax"], est["omega"])))
                previous = grad
            else:
                time += (est["fmax"] + est["fmin"]) / 2.0
        return x, OptimResults(fun=cost, nit=k + 1, nfev=k + 1, message="Number of iterations reached", success=True)


def jacobian_terms(jacobian, nparam):
    """Elastix's ComputeJacobianTerms: TrC, TrCC, maxJJ, maxJCJ, with C = 1/n sum_j J_j^T J_j

    jacobian: array (n, ndim, nparam), or `SparseJacobian` of a B-spline transform (orthonormal basis): then
        C = I (components) x C0, C0 = 1/n sum_j w_j w_j^T, stored by tap offsets (banded)
    """
    sqrt2 = np.sqrt(2.0)
    if isinstance(jacobian, np.ndarray):
        J = jacobian
        n = len(J)
        C = np.einsum("ndp,ndq->pq", J, J) / n
        JJ = np.einsum("ndp,nep->nde", J, J)
        JCJ = np.einsum("ndp,pq,neq->nde", J, C, J)
        maxJJ = np.max(np.einsum("ndp,ndp->n", J, J) + 2 * sqrt2 * np.linalg.norm(JJ, axis=(1, 2)))
        maxJCJ = np.max(np.trace(JCJ, axis1=1, axis2=2) + 2 * sqrt2 * np.linalg.norm(JCJ, axis=(1, 2)))
        return float(np.trace(C)), float(np.sum(C**2)), float(maxJJ), float(maxJCJ)

    flat, weights = jacobian.flat, jacobian.weights
    n, ntap = weights.shape
    ncomp = jacobian.basis.shape[1]
    ncontrol = jacobian.ncontrol
    # tap offsets: flat[:, t2] - flat[:, t1] is the same for all points (taps of a 4^ndim block)
    offsets = flat[0][np.newaxis, :] - flat[0][:, np.newaxis]  # (t1, t2)
    unique, oid = np.unique(offsets, return_inverse=True)
    oid = oid.reshape(ntap, ntap)
    noff = len(unique)
    band = np.zeros(ncontrol * noff)
    for t1 in range(ntap):
        keys = flat[:, t1, np.newaxis] * noff + oid[t1][np.newaxis, :]
        band += np.bincount(keys.ravel(), weights=(weights[:, t1, np.newaxis] * weights).ravel(), minlength=ncontrol * noff)
    band /= n
    band = band.reshape(ncontrol, noff)
    s = np.sum(weights**2, axis=1)  # w_j . w_j
    q = np.zeros(n)  # w_j^T C0 w_j
    for t1 in range(ntap):
        q += weights[:, t1] * np.sum(weights * band[flat[:, t1][:, np.newaxis], oid[t1][np.newaxis, :]], axis=1)
    TrC = ncomp * float(np.sum(s)) / n
    TrCC = ncomp * float(np.sum(band**2))
    maxJJ = float(np.max(s)) * (ncomp + 2 * sqrt2 * np.sqrt(ncomp))
    maxJCJ = float(np.max(q)) * (ncomp + 2 * sqrt2 * np.sqrt(ncomp))
    return TrC, TrCC, maxJJ, maxJCJ


class OptimizerAuto(Optimizer):
    """Default optimizer: "lbfgs" for global transforms, "sgd" (new samples at each iteration) otherwise"""

    def select(self, transform):
        return OptimizerLBFGS() if transform.is_global else OptimizerSGD()

    def optimize(self, objective, init):
        transform = getattr(getattr(objective, "objective", objective), "transform")
        return self.select(transform).optimize(objective, init)

    def __str__(self):
        return "OptimizerAuto"


def lbfgs(func, init, maxiter=100, memory=10, gtol=1e-8, ftol=1e-7, xtol=1e-3, maxstep=1.0, maxls=30):
    """limited-memory BFGS with backtracking (Armijo) line search

    func: returns cost and gradient
    stops when: max |gradient| <= gtol, relative cost decrease <= ftol, or max |step| <= xtol
        (the line search stops backtracking below xtol)
    """
    x = np.asarray(init, dtype=float).copy()
    cost, grad = func(x)
    nfev = 1
    corrections = []  # (s, y, rho)
    res = {"success": False, "message": "Maximum number of iterations reached"}
    nit = 0
    for nit in range(1, maxiter + 1):
        if not np.isfinite(cost):
            res["message"] = "Non-finite cost"
            break
        if np.max(np.abs(grad)) <= gtol:
            res.update(success=True, message="Small gradient")
            break

        # search direction: two-loop recursion
        q = np.array(grad, dtype=float)
        alphas = []
        for s, y, rho in reversed(corrections):
            alpha = rho * (s @ q)
            q -= alpha * y
            alphas.append(alpha)
        if corrections:
            s, y, _ = corrections[-1]
            q *= (s @ y) / (y @ y)
        else:
            q *= maxstep / max(np.max(np.abs(grad)), 1e-300)
        for (s, y, rho), alpha in zip(corrections, reversed(alphas)):
            q += s * (alpha - rho * (y @ q))
        direction = -q
        slope = grad @ direction
        if slope >= 0:
            # not a descent direction: steepest descent, reset memory
            corrections = []
            direction = -grad * maxstep / max(np.max(np.abs(grad)), 1e-300)
            slope = grad @ direction

        # backtracking line search
        step = 1.0
        maxdir = np.max(np.abs(direction))
        accepted = False
        for _ in range(maxls):
            if step * maxdir <= xtol:
                break
            xnew = x + step * direction
            cnew, gnew = func(xnew)
            nfev += 1
            if np.isfinite(cnew) and cnew <= cost + 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            res.update(success=True, message="Small step")
            break

        s, y = xnew - x, gnew - grad
        if s @ y > 1e-12 * (y @ y):
            corrections.append((s, y, 1 / (s @ y)))
            corrections = corrections[-memory:]
        decrease = cost - cnew
        x, cost, grad = xnew, cnew, gnew
        LOGGER.debug(f"iter: {nit}, cost: {cost:3.3}, step: {step}")
        if decrease <= ftol * max(abs(cost), 1e-300):
            res.update(success=True, message="Small cost decrease")
            break

    res.update(nit=nit, nfev=nfev, fun=cost, jac=grad)
    return x, res


class OptimResults(SimpleNamespace):
    """dataclass for optimization results"""

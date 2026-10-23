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
    """Stochastic gradient descent with new samples at each iteration (as Elastix's stochastic optimizers)

    Iteration k moves the (scaled) parameters by -gain_k * gradient / max(|gradient|, g0), with
    gain_k = a / (t_k + 1 + A)^alpha, |.| the largest component and g0 that of the first gradient: no step is larger
    than gain_k (~ mm), and the steps shrink when the gradient becomes smaller than at the start. New samples at
    each iteration avoid fitting the transform to a fixed set of samples (needed for transforms with many
    parameters, e.g. B-splines); fixed number of iterations.

    niter: number of iterations (per pyramid level)
    a, A, alpha: gain sequence (A: stability constant, alpha: decay exponent; 0.602 as in Elastix)
    adaptive: adaptive time t_k (as Elastix's AdaptiveStochasticGradientDescent, Klein et al. 2009):
        t_{k+1} = max(0, t_k + f(-cos(g_k, g_{k-1}))), with the sigmoid f from fmin (< 0) to fmax (> 0), f(0) = 0
        and width omega: the gain decreases when consecutive gradients oppose each other (oscillation around the
        optimum) and increases again when they agree (still far); False: t_k = k. On the synthetic B-spline
        benchmark (known deformation), the fixed decay is more accurate (MI 1.14 vs 1.22 mm, MSE 0.37 vs 0.52 mm)
    """

    def __init__(self, niter=256, a=4.0, A=20, alpha=0.602, adaptive=False, omega=0.1, fmax=1.0, fmin=-0.8):
        self.niter, self.a, self.A, self.alpha = niter, a, A, alpha
        self.adaptive, self.omega, self.fmax, self.fmin = bool(adaptive), omega, fmax, fmin

    def sigmoid(self, x):
        """Klein's sigmoid: fmin at -inf, 0 at 0, fmax at +inf"""
        return self.fmin + (self.fmax - self.fmin) / (1 - (self.fmax / self.fmin) * np.exp(-x / self.omega))

    def optimize(self, objective, init):
        x = np.asarray(init, dtype=float).copy()
        cost, norm, previous, time = np.nan, None, None, 0.0
        for k in range(self.niter):
            if k:
                objective.resample()
            cost, grad = objective(x, gradient=True)
            current = np.max(np.abs(grad))
            if norm is None:
                norm = current
            if not np.isfinite(cost) or max(current, norm) == 0:
                break
            x -= self.a / (time + 1 + self.A) ** self.alpha * grad / max(current, norm)
            if not self.adaptive:
                time = k + 1.0
            elif previous is not None:
                cosine = grad @ previous / max(np.linalg.norm(grad) * np.linalg.norm(previous), 1e-300)
                time = max(0.0, time + float(self.sigmoid(-cosine)))
            previous = grad
        return x, OptimResults(fun=cost, nit=k + 1, nfev=k + 1, message="Number of iterations reached", success=True)


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

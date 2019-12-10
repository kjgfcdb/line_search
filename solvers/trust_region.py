import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
from functions import Q_func


def hebden(Q: Q_func, delta):
    d_0 = -np.linalg.inv(Q.G).dot(Q.g)
    I = np.eye(len(Q.g))
    if norm(d_0) <= delta:
        return d_0
    nu = 0
    epsilon = 1e-2
    while True:
        G_inv = np.linalg.inv(Q.G + nu * I)
        d_nu = -G_inv.dot(Q.g)
        d_nu_grad = -G_inv.dot(d_nu)
        phi_nu = norm(d_nu) - delta
        if np.abs(phi_nu) < epsilon * delta:
            break
        phi_nu_grad = d_nu.dot(d_nu_grad) / norm(d_nu)
        nu = nu - ((phi_nu + delta) * phi_nu) / (phi_nu_grad * delta)
    return d_nu


def cauthy(Q: Q_func, delta):
    d_SD = -delta * Q.g / norm(Q.g)
    gGg = Q.g.dot(Q.G).dot(Q.g)
    if gGg <= 0:
        tau = 1
    else:
        tau = norm(Q.g) ** 3 / (delta * gGg)
        tau = min(1, tau)
    return tau * d_SD


class TrustRegion:
    """信赖域方法求解器"""

    def __init__(self, d_solver, epsilon, delta):
        self._d_solver = d_solver
        self._epsilon = epsilon
        self._delta = delta
        self._iter = 0

    def __call__(self, func, x0, **kwargs):
        f_prev = -np.inf
        bar = tqdm()
        f, g, G = func(x0)
        nf, ng, nG = None, None, None
        updated = False  # 节省一些计算量
        while np.abs(f - f_prev) > self._epsilon:
            if updated:
                f, g, G = nf, ng, nG
                updated = False
            else:
                f, g, G = func(x0)
            f_prev = f
            Q = Q_func(f, g, G)
            d = self._d_solver(Q, self._delta)
            nf, ng, nG = func(x0 + d)
            delta_f = f - nf
            delta_q = Q(np.zeros(len(d))) - Q(d)
            gamma = delta_f / delta_q
            f = nf
            if gamma < 0.25:
                self._delta = self._delta / 4
            elif gamma > 0.75 and np.abs(norm(d) - self._delta) < self._epsilon:
                self._delta = self._delta * 2
            if gamma <= 0:  # 如果gamma<=0，并不会直接退出，因为我们f已经更新了，因此相当于再给了一次机会求解
                continue
            x0 = x0 + d
            updated = True
            bar.desc = f"f:{f}"
            bar.update()
        return x0, f, g

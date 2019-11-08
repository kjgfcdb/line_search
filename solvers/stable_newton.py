import numpy as np

from decomp import m_cholesky
from functions import Phi_func


def stable_newton(func, x0, line_search_func, **kwargs):
    # Gill Murray等人提出的稳定牛顿法
    epsilon = kwargs['epsilon']
    safe_guard = kwargs['safe_guard'] if "safe_guard" in kwargs else None
    g_epsilon = 1e-8
    f_hist = None
    k = 0
    while True:
        f, g, G = func(x0)
        if f_hist is not None:
            if f > f_hist and kwargs['line_search_method'] != 'gll':
                break
            if np.abs(f - f_hist) < epsilon:
                break
        f_hist = f
        if safe_guard is not None and k > safe_guard:
            break
        L, D, E = m_cholesky(G)
        if np.linalg.norm(g) >= g_epsilon:
            d = np.linalg.solve(L.dot(D).dot(L.T), -g)
        else:
            xi = np.zeros(E.shape[0])
            for i in range(E.shape[0]):
                xi[i] = D[i][i] - E[i][i]
            t = np.argmin(xi)
            if xi[t] >= 0:
                break
            else:
                eye = np.zeros(E.shape[0])
                eye[t] = 1
                d = np.linalg.solve(L.T, eye)
                if g.dot(d) > 0:
                    d = -d
        phi = Phi_func(func, x0, d)
        alpha = line_search_func(phi, safe_guard=safe_guard)
        x0 = x0 + alpha * d
        k += 1
        print(f"Stable Newton: Epoch: {k}\t function value: {f}")

    return x0, f, g

import numpy as np
from functions import Phi_func


def damp_newton(func, x0, line_search_func, **kwargs):
    epsilon = kwargs['epsilon']
    safe_guard = kwargs['safe_guard'] if 'safe_guard' in kwargs else None
    f_hist = None
    k = 0
    while True:
        f, g, G = func(x0)
        if safe_guard is not None and k > safe_guard:
            break
        if f_hist is not None and np.abs(f_hist - f) < epsilon:
            break
        f_hist = f
        d = np.linalg.solve(G, -g)
        phi = Phi_func(func, x0, d)
        alpha = line_search_func(phi, **kwargs)
        x0 = x0 + alpha * d
        k += 1
        print(f"Damp Newton: Epoch: {k}\t function value: {f}")
    return x0, f, g

import numpy.linalg as la
import numpy as np
from collections import deque

hist = None


def l_bfgs(func, init, m):
    global hist
    if hist is None:
        hist = deque(maxlen=m)
    eps = 1e-5

    while True:
        f, g, G = func(init)
        if la.norm(g) < eps * max(1, la.norm(init)):
            break
        H = np.eye(G.shape[0])
        q = g
        alpha_list = []
        # two loops
        for s, y, rho in reversed(hist):
            alpha = rho * s.dot(q)
            q = q - alpha * y
            alpha_list.append(alpha)
        r = H.dot(q)
        alpha_list = alpha_list.reverse()
        for (s, y, rho), alpha in zip(hist, alpha_list):
            beta = rho * y.dot(r)
            r = r + s * (alpha - beta)

        d = -r

        init = init + d

    return init


def compact_l_bfgs(func, init, m):
    global hist
    if hist is None:
        hist = deque(maxlen=m)
    eps = 1e-5
    while True:
        f, g, G = func(init)
        if la.norm(g) < eps * max(1, la.norm(init)):
            break

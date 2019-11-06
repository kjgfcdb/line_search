from itertools import product

import numpy as np

from decomp import bunch_parlett
from functions import Phi_func


def _get_d(g, G):
    L, D = bunch_parlett(G)
    eig, _ = np.linalg.eig(D)
    if all(i > 0 for i in eig):
        d = - np.linalg.inv(L.T).dot(np.linalg.inv(D)).dot(np.linalg.inv(L)).dot(g)
    else:
        D_hat = np.zeros(D.shape)
        for i in range(D.shape[0]):
            if i + 1 < D.shape[0] and D[i][i + 1] != 0:
                temp = D[i:i + 2, i:i + 2]
                w, v = np.linalg.eig(temp)
                w_hat = np.zeros(w.shape)
                for j in range(2):
                    if w[j] > 0:
                        w_hat[j] = 1 / w[j]
                block = v.dot(np.diag(w_hat)).dot(v.T)
                for j, k in product(range(2), range(2)):
                    D_hat[i + j][i + k] = block[j][k]
            else:
                if D[i][i] > 0:
                    D_hat[i][i] = 1 / D[i][i]
        d = - np.linalg.inv(L.T).dot(D_hat).dot(np.linalg.inv(L)).dot(g)
        if all(item == 0 for item in d):
            d = np.linalg.solve(G, 0)
            if g.dot(d) > 0:
                d = -d
    return d


def fletcher_freeman(func, x0, **kwargs):
    epsilon = kwargs['epsilon']
    line_search_func = kwargs['line_search_func']
    k = 0
    f_hist = None
    while True:
        f, g, G = func(x0)
        if k > 200:
            break
        if f_hist is not None and np.abs(f_hist - f) < epsilon:
            break
        f_hist = f
        d = _get_d(g, G)
        phi = Phi_func(func, x0, d)
        alpha = line_search_func(phi, safe_guard=True)
        x0 = x0 + alpha * d
        k += 1
        print(f"Epoch: {k}\t function value: {f}")
    return x0, f, g

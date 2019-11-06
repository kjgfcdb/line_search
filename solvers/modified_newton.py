import numpy as np

from decomp import m_cholesky


def modified_newton(func, x0, **kwargs):
    # Goldfeld等人提出的修正牛顿法
    epsilon = kwargs['epsilon']
    f_hist = None
    k = 0
    while True:
        f, g, G = func(x0)
        if f_hist is not None and np.abs(f - f_hist) < epsilon:
            break
        f_hist = f
        if k > 40:
            break
        L, D, E = m_cholesky(G)
        if all(E[i, i] == 0 for i in range(E.shape[0])):
            d = np.linalg.solve(G, -g)
        else:
            b1 = float('inf')
            for i in range(G.shape[0]):
                s = sum(np.abs(G[i][j]) for j in range(G.shape[0]) if j != i)
                b1 = min(np.abs(G[i][i] - s), b1)
            b2 = min(E[i][i] for i in range(E.shape[0]))
            vk = min(b1, b2)
            G_hat = L.dot(D).dot(L.T) + vk * np.eye(E.shape[0])
            d = np.linalg.solve(G_hat, -g)
        x0 = x0 + d
        k += 1
        print(f"Epoch: {k}\t function value: {f}")

    return x0, f, g

import numpy as np


def m_cholesky(G):
    # init
    delta = 1e-5
    n = G.shape[0]
    L = np.zeros((n, n))
    D = np.zeros((n, n))
    E = np.zeros((n, n))
    C = np.zeros((n, n))
    nu = max(np.sqrt(n ** 2 - 1), 1)
    gamma = float('-inf')
    for i in range(n):
        gamma = max(gamma, np.abs(G[i][i]))
    xi = float('-inf')
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi = max(xi, np.abs(G[i][j]))
    beta_2 = max([gamma, xi / nu])
    for i in range(n):
        C[i][i] = G[i][i]
    j = 0
    # loop
    while True:
        c_ii = [np.abs(C[i][i]) for i in range(j, n)]
        q = np.argmax(c_ii) + j
        G[[q, j], :] = G[[j, q], :]
        G[:, [q, j]] = G[:, [j, q]]
        for s in range(j):
            L[j][s] = C[j][s] / D[s][s]
        for i in range(j + 1, n):
            temp = 0
            for s in range(j):
                temp += L[j][s] * C[i][s]
            C[i][j] = G[i][j] - temp
        if j == n - 1:
            theta_j = 0
        else:
            theta_j = max(np.abs(C[i][j]) for i in range(j + 1, n))
        D[j][j] = max([delta, np.abs(C[j][j]), theta_j ** 2 / beta_2])
        E[j][j] = D[j][j] - C[j][j]
        if j == n - 1:
            break
        for i in range(j + 1, n):
            C[i][i] -= C[i][j] ** 2 / D[j][j]
        j += 1
    for i in range(n):
        L[i][i] = 1
    return L, D, E


def test(A, B):
    C = A - B
    F = (C ** 2).sum()
    return F


if __name__ == "__main__":
    G = np.array([
        [1, 1, 2],
        [1, 1 + 20 ** (-20), 3],
        [2, 3, 1]
    ])
    L, D, E = m_cholesky(G.copy())

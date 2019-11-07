import numpy as np
from itertools import product


def m_cholesky(G):
    # init
    delta = 1e-8
    n = G.shape[0]
    L = np.zeros((n, n))
    D = np.zeros((n, n))
    E = np.zeros((n, n))
    C = np.zeros((n, n))
    nu = max(np.sqrt(n ** 2 - 1), 1)
    gamma = max(np.abs(G[i][i]) for i in range(n))
    xi = max(np.abs(G[i][j]) for i,j in product(range(n), range(n)) if i!=j)
    beta_2 = max([gamma, xi / nu])
    for i in range(n):
        C[i][i] = G[i][i]
    j = 0
    # loop
    while True:
        c_ii = [np.abs(C[i][i]) for i in range(j, n)]
        q = np.argmax(c_ii) + j
        # G[[q, j], :] = G[[j, q], :]
        # G[:, [q, j]] = G[:, [j, q]]
        for s in range(j):
            L[j][s] = C[j][s] / D[s][s]
        for i in range(j + 1, n):
            C[i][j] = G[i][j] - sum(L[j][s]*C[i][s] for s in range(j))
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


if __name__ == "__main__":
    for i in range(10):
        m = np.random.randn(12,12)
        m = m.dot(m.T)
        m_old = m.copy()
        L, D, E = m_cholesky(m)
        A = L.dot(D).dot(L.T)
        B = m_old + E
        # print(np.linalg.norm(A-B))
        print(np.allclose(A,B))

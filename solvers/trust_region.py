import numpy as np
from tqdm import tqdm
from numpy.linalg import norm, eigvals, inv, eig
from functions import Q_func


def hebden(Q: Q_func, delta):
    """Hebden迭代算法
    
    Parameters
    ----------
    Q : Q_func
        要求解的Q函数
    delta : float
        信赖域半径
    
    Returns
    -------
    d_nu：求解得到的信赖域方向
    """
    d_0 = -np.linalg.inv(Q.G).dot(Q.g)
    I = np.eye(len(Q.g))
    if norm(d_0) <= delta:
        return d_0
    epsilon = 1e-6
    min_eigval = min(eigvals(Q.G))
    nu = max(-1.02 * min_eigval, epsilon)
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
    """柯西点算法
    
    Parameters
    ----------
    Q : Q_func
        要求解的Q函数
    delta : float
        信赖域半径
    
    Returns
    -------
    d：求解得到的信赖域方向
    """
    d_SD = -delta * Q.g / norm(Q.g)
    gGg = Q.g.dot(Q.G).dot(Q.g)
    if gGg <= 0:
        tau = 1
    else:
        tau = norm(Q.g) ** 3 / (delta * gGg)
        tau = min(1, tau)
    return tau * d_SD


def span(s1, s2, g, G, delta):
    """
    求解
        min g^T x + 0.5 * x^T G x
        s.t. norm(x) <= delta
    且x为由s1和s2张成的子空间中的一个向量，即
        x = a * s1 + b * s2

    Parameters
    ----------
    s1 : np.ndarray
        第一个向量
    s2 : np.ndarray
        第二个向量
    g : np.ndarray
        当前点的梯度向量
    G : np.ndarray
        当前点的Hesse矩阵
    delta : float
        信赖域半径
    
    Returns
    -------
    x：求解上述函数得到的解
    """
    # 构造M
    m1 = s1
    m2 = s2 - s2.dot(s1) * s1 / (s1.dot(s1))
    m1 = m1 / norm(m1)
    m2 = m2 / norm(m2)
    M = np.hstack((m1.reshape(len(m1), 1), m2.reshape(len(m2), 1)))  # m x2

    A = np.matmul(np.matmul(M.T, G), M)
    B = g.dot(M)

    a1 = A[0][0]
    a2 = A[0][1]
    a3 = A[1][1]
    a4 = B[0]
    a5 = B[1]

    c1 = delta * (-a2 * delta + a4)
    c2 = 2 * delta * ((a3 - a1) * delta + a5)
    c3 = 6 * delta ** 2 * a2
    c4 = c2
    c5 = -a2 * delta ** 2 - a4 * delta

    roots = np.roots([c1, c2, c3, c4, c5])
    final_q = None
    corr_val = np.inf
    for root in roots:
        if not np.isreal(root):
            continue
        root = np.real(root)
        x = 2 * delta * root / (1 + root ** 2)
        y = delta * (1 - root ** 2) / (1 + root ** 2)
        q = np.array([x, y])
        cur_func_val = B.dot(q) + 1 / 2 * q.dot(A).dot(q)
        if cur_func_val < corr_val:
            corr_val = cur_func_val
            final_q = q
    return M.dot(final_q)


def two_d_subspace_min(Q: Q_func, delta):
    """二维子空间极小化算法
    
    Parameters
    ----------
    Q : Q_func
        要求解的Q函数
    delta : float
        信赖域半径
    
    Returns
    -------
    d：求解得到的信赖域方向
    """
    tol = 1e-10
    vals, vectors = eig(Q.G)
    v = min(vals)
    if v < 0:
        v = 1.5 * v
    alpha = np.abs(v)
    if v > tol:  # 正定矩阵
        d = -inv(Q.G).dot(Q.g)
        if norm(d) <= delta:
            return d
        return span(Q.g, d, Q.g, Q.G, delta)
    elif alpha < tol:  # 存在0特征值
        return cauthy(Q, delta)
    else:  # 存在负特征值
        d = -inv(Q.G + alpha * np.eye(len(Q.g))).dot(Q.g)
        if norm(d) <= delta:
            q = vectors[:, np.argmin(vals)]
            q = q / norm(q)
            gamma = -d.dot(q) + np.sqrt((d.dot(q)) ** 2 + delta ** 2 - d.dot(d))
            d = d + gamma * q
            return d
        else:
            return span(Q.g, d, Q.g, Q.G, delta)


class TrustRegion:
    """信赖域方法求解器"""

    def __init__(self, d_solver, epsilon, delta):
        self._d_solver = d_solver
        self._epsilon = epsilon
        self._delta = delta
        self.iters = 0
        self.time = 0

    def __call__(self, func, x0, **kwargs):
        """调用信赖域方法求解
        
        Parameters
        ----------
        func : Evaluater
            要求解的函数，调用返回函数值、梯度、Hesse矩阵
        x0 : list
            优化函数的初始值
        
        Returns
        -------
        x0：最优解
        f：最优解对应的函数值
        g：最优解对应的函数梯度
        """
        f_prev = -np.inf
        bar = tqdm()
        f, g, G = func(x0)
        nf, ng, nG = None, None, None
        updated = False  # 节省一些计算量
        safe_guard = kwargs['safe_guard'] if 'safe_guard' in kwargs else None
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
            elif gamma > 0.75 and norm(d) == self._delta:  # np.abs(norm(d) - self._delta) < self._epsilon:
                self._delta = min(self._delta * 2, 1)
            if gamma <= 0:  # 如果gamma<=0，并不会直接退出，因为我们f已经更新了，因此相当于再给了一次机会求解
                continue
            x0 = x0 + d
            updated = True
            bar.desc = f"f:{f}"
            bar.update()
            if safe_guard is not None and bar.n >= safe_guard:
                break
        bar.close()
        self.iters = bar.n
        self.time = bar.last_print_t - bar.start_t
        return x0, f, g

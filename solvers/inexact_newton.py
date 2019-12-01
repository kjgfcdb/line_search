"""非精确牛顿法实现
"""
import numpy as np
import numpy.linalg as la
from scipy.optimize import fminbound
from scipy.sparse.linalg import gmres
from tqdm import tqdm
from functions import Phi_func
from step_size import armijo_goldstein_linesearch


def inexact_newton(func, init, **kwargs):
    """用非精确牛顿法求解优化问题
    
    Parameters
    ----------
    func : Evaluater
        函数对象，可以通过调用返回函数值、梯度、Hessian矩阵
    init : list
        初始点坐标
    
    Returns
    -------
    init: np.ndarray
        最优解
    f: float
        最优解对应的函数值
    g: np.ndarray
        最优解对应的梯度
    
    Raises
    ------
    NotImplementedError
        如果选择的choice不是1或者2，那么报错
    """
    eta_max = kwargs['eta_max'] if 'eta_max' in kwargs else 0.9
    t = kwargs['t'] if 't' in kwargs else 1e-4
    theta_min = kwargs['theta_min'] if 'theta_min' in kwargs else 0.1
    theta_max = kwargs['theta_max'] if 'theta_max' in kwargs else 0.5
    eps = kwargs['eps'] if 'eps' in kwargs else 1e-5
    assert 'choice' in kwargs, "必须选择一种非精确牛顿法的策略！"
    choice = kwargs['choice']

    g_prev = None
    G_prev = None
    f_prev = None
    bar = tqdm()
    while True:
        f, g, G = func(init)
        # if f_prev is not None and f > f_prev:
        #     break
        if la.norm(g) < eps * max(1, la.norm(init)):
            break
        if choice == 1:
            if g_prev is None:
                eta_k = 0.5
            else:
                eta_k = np.abs(la.norm(g) - la.norm(g_prev +
                                                    G_prev.dot(d_k))) / la.norm(g_prev)
        elif choice == 2:
            gamma = kwargs['gamma'] if 'gamma' in kwargs else 1
            alpha = kwargs['alpha'] if 'alpha' in kwargs else (1+np.sqrt(5))/2
            if g_prev is None:
                eta_k = 0.5
            else:
                eta_k_old = eta_k
                eta_k = gamma * (la.norm(g) / la.norm(g_prev)) ** alpha
                if gamma * eta_k_old**alpha > 0.1:
                    eta_k = max(eta_k, gamma * eta_k_old ** alpha)
        else:
            raise NotImplementedError()
        eta_k = min(eta_k, eta_max)
        # d_k, exit_code = gmres(G,  -g, restart=20, tol=eta_k, maxiter=10)
        d_k, exit_code = gmres(G,  (eta_k-1)*g, maxiter=10, restart=20)
        if la.norm(d_k) == 0:
            break

        phi = Phi_func(func, init, d_k)
        alpha = armijo_goldstein_linesearch(phi, safe_guard=20)
        init = init + d_k * alpha
        g_prev = g
        G_prev = G
        f_prev = f
        bar.desc = '函数值:' + str(f)
        bar.update()
    bar.close()
    inexact_newton.iters = bar.n
    return init, f, g

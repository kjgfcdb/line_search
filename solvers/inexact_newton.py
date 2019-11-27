"""非精确牛顿法实现
"""
import numpy as np
import numpy.linalg as la
from scipy.optimize import fminbound
from scipy.sparse.linalg import gmres
from tqdm import tqdm
from functions import Phi_func
from step_size import armijo_goldstein_linesearch


def get_theta(g_k, d_k, g_one, G, theta_min, theta_max):
    c = g_k.dot(g_k)
    # b = 2 * g_k.dot(d_k)
    b = 2 * d_k.dot(G).dot(g_k)
    a = g_one.dot(g_one) - b - c

    def temp_func(x): return a * x ** 2 + b * x + c

    return fminbound(temp_func, theta_min, theta_max)


def inexact_newton(func, init, **kwargs):
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

        # while la.norm(func(init + d_k, g_only=True)) > (1 - t * (1 - eta_k)) * la.norm(g):
        #     g_one = func(init + d_k, g_only=True)
        #     theta = get_theta(g, d_k, g_one, G, theta_min, theta_max)
        #     d_k = theta * d_k
        #     eta_k = 1 - theta * (1 - eta_k)

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
    inexact_newton.time = bar.last_print_t - bar.start_t
    return init, f, g

from collections import deque

import numpy as np
import numpy.linalg as la
from tqdm import tqdm
from step_size import armijo_goldstein_linesearch
from functions import Phi_func


def l_bfgs_two_loop(hist, H0, g):
    q = g
    alpha_list = []
    for s, y, rho in reversed(hist):
        alpha = rho * s.dot(q)
        q = q - alpha * y
        alpha_list.append(alpha)
    r = H0.dot(q)
    alpha_list.reverse()
    for (s, y, rho), alpha in zip(hist, alpha_list):
        beta = rho * y.dot(r)
        r = r + s * (alpha - beta)
    return r


def l_bfgs(func, init, **kwargs):
    assert 'm' in kwargs, "必须指定LBFGS方法的队列长度！"
    m = kwargs['m']
    hist = deque(maxlen=m)
    eps = 1e-5

    g_prev = None
    init_prev = None
    bar = tqdm()
    while True:
        f, g, G = func(init)
        if g_prev is not None and init_prev is not None:
            y = g - g_prev
            s = init - init_prev
            rho = 1 / y.dot(s)
            hist.append((s, y, rho))

        if la.norm(g) < eps * max(1, la.norm(init)):
            break
        H0 = np.eye(G.shape[0])
        r = l_bfgs_two_loop(hist, H0, g)
        d = -r

        phi = Phi_func(func, init, d)
        alpha = armijo_goldstein_linesearch(phi, safe_guard=200)

        g_prev = g
        init_prev = init.copy()
        init = init + alpha * d
        bar.desc = '函数值:' + str(f)
        bar.update()
    return init, f


def compact_l_bfgs(func, init, **kwargs):
    assert 'm' in kwargs, "必须指定Compact LBFGS方法的队列长度！"
    m = kwargs['m']
    hist = deque(maxlen=m)
    eps = 1e-5
    bar = tqdm()
    g_prev = None
    init_prev = None
    while len(hist) < m:
        f, g, G = func(init)
        if g_prev is not None and init_prev is not None:
            s = init - init_prev
            y = g - g_prev
            rho = 1 / y.dot(s)
            hist.append((s, y, rho))
        H0 = np.eye(G.shape[0])
        d = -l_bfgs_two_loop(hist, H0, g)
        phi = Phi_func(func, init, d)
        alpha = armijo_goldstein_linesearch(phi, safe_guard=200)
        g_prev = g
        init_prev = init.copy()
        init = init + alpha * d
        bar.desc = '函数值:' + str(f)
        bar.update()
    # construct S_{k-1}, Y_{k-1}, R_{k-1}, D_{k-1}
    ss = []
    yy = []
    for (s, y, rho) in hist:
        ss.append(s)
        yy.append(y)
    S = np.vstack(ss).T
    Y = np.vstack(yy).T
    R = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i > j:
                continue
            R[i][j] = ss[i].dot(yy[j])
    D = [ss[i].dot(yy[i]) for i in range(m)]
    while True:
        f, g, G = func(init)
        s = init - init_prev
        y = g - g_prev
        rho = 1 / y.dot(s)
        hist.append((s, y, rho))
        if la.norm(g) < eps * max(1, la.norm(init)):
            break
        # update S_k, Y_k
        S = np.hstack((S[:, 1:], s.reshape(len(s), 1)))
        Y = np.hstack((Y[:, 1:], y.reshape(len(y), 1)))
        # calculate S_k.T.dot(g_k), Y_k.T.dot(g_k)
        Sg = S.T.dot(g)
        Yg = Y.T.dot(g)
        # update R_k, Y_k.T.dot(Y_k), D_k
        R = np.vstack((R[1:, 1:], np.zeros((1, m - 1))))
        R = np.hstack((R, S.T.dot(y).reshape(m, 1)))
        YY = Y.T.dot(Y)
        sy = s.dot(y)
        D = D[1:] + [s.dot(y)]
        D_mat = np.diag(D)
        # compute gamma
        gamma = sy / y.dot(y)
        # get p
        R_inv = la.inv(R)
        p = R_inv.T.dot(D_mat + gamma * YY).dot(R_inv).dot(Sg) - gamma * R_inv.T.dot(Yg)
        p = np.concatenate((p, -R_inv.dot(Sg)))
        # compute H_k g_k
        r = gamma * g + np.hstack((S, gamma * Y)).dot(p)

        d = -r

        phi = Phi_func(func, init, d)
        alpha = armijo_goldstein_linesearch(phi, safe_guard=100)
        g_prev = g
        init_prev = init.copy()
        init = init + alpha * d

        bar.desc = '函数值:' + str(f)
        bar.update()
    return init, f

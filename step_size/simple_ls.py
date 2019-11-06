import numpy as np
from random import random
from conditions import armijo_goldstein


def simple_linesearch(phi, rho=0.25, lb=0.1, ub=0.9, minstep=1e-4, **kwargs):
    # TODO: 此处实现二次插值
    assert 0 < rho < 0.5
    assert 0 < lb < ub < 1
    safe_guard = kwargs['safe_guard'] if 'safe_guard' in kwargs else None
    stepsize = 1
    cnt = 0
    while True:
        if safe_guard is not None and cnt > safe_guard:
            break
        cnt += 1
        f, g = phi(0)
        f_new, _ = phi(stepsize)
        cond = armijo_goldstein(f_new, f, stepsize, g, None, rho)
        if not cond[0]:
            w = random() * (ub - lb) + lb
            stepsize = w * stepsize
        else:
            break
        if np.abs(stepsize) < minstep:
            break
    return stepsize

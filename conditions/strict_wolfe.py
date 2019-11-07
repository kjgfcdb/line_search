import numpy as np


def strict_wolfe(f_new, f, alpha, g_new, g, d, rho=0.25, sigma=0.75):
    """
    Check if the wolfe condition is statisfied, i.e
        f_new <= f + rho * alpha * g * d
        |g_new * d| <= -sigma * g * d
    output:
        cond: 长度为2的数组，cond[0]为True则表示第一个条件满足，cond[1]为True表示第二个条件满足
    """
    cond = [False, False]
    assert (0 < rho < 0.5) and (
            rho < sigma < 1), "rho in (0, 0.5) and sigma in (rho, 1)"
    if d is None:
        gd = g
        gd_new = g_new
    else:
        gd = g.dot(d)
        gd_new = g_new.dot(d)
    if f_new <= f + rho * alpha * gd:
        cond[0] = True
    if np.abs(gd_new) <= -sigma * gd:
        cond[1] = True
    return cond

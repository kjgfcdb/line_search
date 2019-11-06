from collections import deque
from conditions import gll


def gll_linesearch(phi, f_hist_max, stepsize0=1, rho=0.5, sigma=0.5):
    stepsize = stepsize0
    f, g = phi(0)
    while True:
        f_new, _ = phi(stepsize)
        if not gll(f_new, f_hist_max, stepsize, g, None, rho):
            stepsize = sigma * stepsize
        else:
            break
    return stepsize

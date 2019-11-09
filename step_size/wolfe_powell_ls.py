import numpy as np
from conditions import wolfe, strict_wolfe
from step_size import interp22


def wolfe_powell_linesearch(phi, stepsize=1, rho=0.1, sigma=0.75, use_strict_wolfe=False, **kwargs):
    a = 0
    safe_guard = kwargs['safe_guard'] if 'safe_guard' in kwargs else None
    # b = alpha_max
    assert stepsize > 0
    f, g = phi(0)
    cnt = 0
    while True:
        if safe_guard is not None and cnt > safe_guard:
            break
        if np.isnan(stepsize) or np.isinf(stepsize):
            return 0
        cnt += 1
        f_new, g_new = phi(stepsize)
        if use_strict_wolfe:
            cond = strict_wolfe(f_new, f, stepsize, g_new, g, None, rho, sigma)
        else:
            cond = wolfe(f_new, f, stepsize, g_new, g, None, rho, sigma)
        if not cond[0]:
            alpha_hat = interp22(a, stepsize, phi)
            # b = stepsize
            stepsize = alpha_hat
        else:
            if not cond[1]:
                alpha_hat = stepsize + (stepsize - a) * g_new / (g - g_new)
                a = stepsize
                f = f_new
                g = g_new
                stepsize = alpha_hat
            else:
                break
    return stepsize

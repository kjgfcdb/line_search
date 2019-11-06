from conditions import armijo_goldstein


def armijo_goldstein_linesearch(phi, stepsize=1, rho=0.25, t=1.2, alpha_max=float('inf'), **kwargs):
    a = 0
    b = alpha_max
    safe_guard = kwargs['safe_guard'] if 'safe_guard' in kwargs else None
    assert t > 1
    assert 0 < rho < 0.5
    assert alpha_max >= 0
    v, g = phi(0)
    cnt = 0
    while True:
        v_next, g_next = phi(stepsize)
        cnt += 1
        if safe_guard is not None and cnt > safe_guard:
            break
        cond = armijo_goldstein(v_next, v, stepsize, g, None, rho)
        if cond[0]:
            if cond[1]:
                break
            else:
                a = stepsize
                if b < float('inf'):
                    stepsize = (a + b) / 2
                else:
                    stepsize = t * stepsize
        else:
            b = stepsize
            stepsize = (a + b) / 2
    return stepsize

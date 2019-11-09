from conditions import armijo_goldstein


def armijo_goldstein_linesearch(phi, stepsize=1, rho=0.1, t=1.2, **kwargs):
    """Armijo Goldstein线搜索准则
    
    Parameters
    ----------
    phi : 函数
        定义的phi函数，phi(alpha) = f(x0 + alpha * d)
    stepsize : float
        初始步长
    rho : float
        Armijo Goldstein准则中的\rho
    t : float
        Armijo Goldstein准则中对满足第一个条件但是不满足第二个条件的alpha进行缩放的系数
    Returns
    -------
    返回搜索得到的步长
    """
    a = 0
    b = float('inf')
    safe_guard = kwargs['safe_guard'] if 'safe_guard' in kwargs else None
    assert t > 1
    assert 0 < rho < 0.5
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

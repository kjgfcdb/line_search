def interp22(stepsize1, stepsize2, phi):
    """两点二次插值法
    
    Parameters
    ----------
    stepsize1 : float
        alpha_1
    stepsize2 : float
        alpha_2
    phi : 函数
        定义的phi函数
    
    Returns
    -------
    返回下一个步长
    """
    v1, g1 = phi(stepsize1)
    v2, _ = phi(stepsize2)
    nxt_step_size = stepsize1 - 0.5 * \
                    ((stepsize1 - stepsize2) * g1) / (g1 - (v1 - v2) / (stepsize1 - stepsize2))
    return nxt_step_size

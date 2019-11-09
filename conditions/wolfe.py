def wolfe(f_new, f, alpha, g_new, g, d, rho=0.25, sigma=0.75):
    """
    判断是否满足Wolfe条件
        f_new <= f + rho * alpha * g * d
        g_new * d >= sigma * g * d
    input:
        f_new:新函数值
        f:旧函数值
        alpha:步长
        g_new:新的梯度值
        g:旧梯度值
        d:下降方向
        rho:Wolfe准则中的\rho
        sigma:Wolfe准则中的\sigma
    output:
        cond: 长度为2的数组，cond[0]为True则表示第一个条件满足，cond[1]为True表示第二个条件满足
    """
    cond = [False, False]
    assert (0 < rho < 0.5) and (rho < sigma < 1), "rho in (0, 0.5) and sigma in (rho, 1)"
    if d is None:
        gd = g
        gd_new = g_new
    else:
        gd = g.dot(d)
        gd_new = g_new.dot(d)
    if f_new <= f + rho * alpha * gd:
        cond[0] = True
    if gd_new >= sigma * gd:
        cond[1] = True
    return cond

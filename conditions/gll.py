def gll(f_new, f_hist_max, stepsize, g, d, rho):
    """判断输入是否满足GLL准则
    
    Parameters
    ----------
    f_new : 新的函数值
    f_hist_max : 在过去一段迭代过程中，最大的函数值
    stepsize : 步长
    g : 导数
    d : 下降方向
    rho : GLL准则中的\rho
    Returns
    -------
    如果满足GLL准则则返回True，否则返回False
    """
    assert 0 < rho < 1
    if d is not None:
        rhs = f_hist_max + rho * stepsize * g.dot(d)
    else:
        rhs = f_hist_max + rho * stepsize * g
    return f_new <= rhs

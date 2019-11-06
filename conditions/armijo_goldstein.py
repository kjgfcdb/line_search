def armijo_goldstein(f_new, f, alpha, g, d, rho):
    """
     检查是否满足Armijo-goldstein准则，即
        f_new <= f + rho * alpha * g * d
        f_new >= f + (1-rho) * alpha * g * d
    :param f_new: 新函数值
    :param f: 当前函数值
    :param alpha: 待检查的alpha
    :param g: 梯度，向量，如果d不存在那么就是标量
    :param d: 下降方向，向量，如果使用phi函数那么为None
    :param rho: 常系数
    :return:
        cond: 长度为2的数组，cond[0]为True则表示第一个条件满足，cond[1]为True表示第二个条件满足
    """

    cond = [False, False]
    assert 0 < rho < 0.5, "rho must be in (0,0.5)"
    f_delta = f_new - f
    if d is not None:
        rhs = alpha * g.dot(d)
    else:  # 如果d不存在，那么此时用的是函数phi，不需要d，d本身包含在g中
        rhs = alpha * g
    if f_delta <= rho * rhs:
        cond[0] = True
    if f_delta >= (1 - rho) * rhs:
        cond[1] = True
    return cond

def interp22(stepsize1, stepsize2, phi):
    # 两点二次插值法
    # stepsize1: alpha1，通常情况下是alpha_{k}
    # stepsize2: alpha2，通常情况是alpha_{k-1}
    # phi: 函数phi(alpha) = f(x + alpha*d)，要求返回结果为(函数值，导数)形式
    v1, g1 = phi(stepsize1)
    v2, _ = phi(stepsize2)
    nxt_step_size = stepsize1 - 0.5 * \
                    ((stepsize1 - stepsize2) * g1) / (g1 - (v1 - v2) / (stepsize1 - stepsize2))
    return nxt_step_size

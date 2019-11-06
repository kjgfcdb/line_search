def gll(f_new, f_hist_max, stepsize, g, d, rho):
    assert 0 < rho < 1
    if d is not None:
        rhs = f_hist_max + rho * stepsize * g.dot(d)
    else:
        rhs = f_hist_max + rho * stepsize * g
    return f_new <= rhs

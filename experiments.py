from functions import Evaluater
from solvers import damp_newton, fletcher_freeman, stable_newton
from step_size import armijo_goldstein_linesearch, wolfe_powell_linesearch, gll_linesearch, simple_linesearch
from stackprinter import set_excepthook

set_excepthook(style='darkbg2')

LINE_SEARCH_METHODS = {
    "armijo-goldstein": armijo_goldstein_linesearch,
    "wolfe-powell": wolfe_powell_linesearch,
    "gll": gll_linesearch,
    "simple": simple_linesearch
}

SOLVERS = {
    "damp_newton": damp_newton,
    "stable_newton": stable_newton,
    "fletcher_freeman": fletcher_freeman
}


def main(func_name, line_search_method, solver_name, init, **kwargs):
    line_search_func = LINE_SEARCH_METHODS[line_search_method]
    func = Evaluater(func_name, **kwargs)
    solver = SOLVERS[solver_name]
    x, f, g = solver(func, init, line_search_func=line_search_func, epsilon=1e-12, safe_guard=100)
    print(f"x: {x}\tf: {f}\tg: {g}")


if __name__ == '__main__':
    # func_name = "powell_badly_scaled"
    func_name = "biggs_exp6"
    # func_name = "extended_powell_singular"

    # line_search_method = 'armijo-goldstein'
    line_search_method = 'wolfe-powell'
    # line_search_method = 'simple'

    # solver_name = "damp_newton"
    # solver_name = "fletcher_freeman"
    solver_name = "stable_newton"

    m = 13
    # init = [0, 1]
    init = [1,2,1,1,1,1]
    #
    # init = []
    # for i in range(m):
        # if i % 4 == 0:
            # init.append(3)
        # elif i % 4 == 1:
            # init.append(-1)
        # elif i % 4 == 2:
            # init.append(0)
        # else:
            # init.append(1)

    main(func_name, line_search_method, solver_name, init, m=m)

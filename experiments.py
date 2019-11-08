from functions import Evaluater
from solvers import damp_newton, fletcher_freeman, stable_newton
from step_size import armijo_goldstein_linesearch, wolfe_powell_linesearch, gll_linesearch, simple_linesearch
from stackprinter import set_excepthook
import argparse

set_excepthook(style='darkbg2')

LINE_SEARCH_METHODS = {
    "armijo_goldstein": armijo_goldstein_linesearch,
    "wolfe_powell": wolfe_powell_linesearch,
    "gll": gll_linesearch,
    "simple": simple_linesearch
}

SOLVERS = {
    "damp_newton": damp_newton,
    "stable_newton": stable_newton,
    "fletcher_freeman": fletcher_freeman
}


def main(func_name, line_search_method, solver_name, **kwargs):
    line_search_func = LINE_SEARCH_METHODS[line_search_method]
    if line_search_method == "gll":
        line_search_func = line_search_func(**kwargs)
    func = Evaluater(func_name, **kwargs)
    init = func.init
    solver = SOLVERS[solver_name]
    kwargs.update(dict(
        line_search_method = line_search_method,
        epsilon = 1e-15,
        safe_guard = 200
    ))
    x, f, g = solver(func, init, line_search_func, **kwargs)
    print(f"x: {x}\tf: {f}\tg: {g}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimization experiments")
    parser.add_argument(
        "-func",
        type=str,
        help="The function you want to optimize",
        default="powell_badly_scaled"
    )
    parser.add_argument(
        "-ls",
        type=str,
        help="The line search method you want to use",
        default="armijo_goldstein"
    )
    parser.add_argument(
        "-solver",
        type=str,
        help="The solver that solves this optimization problem",
        default="damp_newton"
    )
    parser.add_argument(
        "-m",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "-count",
        dest='count',
        action='store_true'
    )
    parser.set_defaults(count=False)
    args = parser.parse_args()
    func_name = args.func
    line_search_method = args.ls
    solver_name = args.solver
    m = args.m
    count = args.count
    # func_name = "powell_badly_scaled"
    # func_name = "biggs_exp6"
    # func_name = "extended_powell_singular"

    # line_search_method = 'armijo_goldstein'
    # line_search_method = 'wolfe_powell'
    # line_search_method = 'simple'
    # line_search_method = 'gll'

    # solver_name = "damp_newton"
    # solver_name = "fletcher_freeman"
    # solver_name = "stable_newton"
    print(args)

    main(func_name, line_search_method, solver_name, m=m, count=count)

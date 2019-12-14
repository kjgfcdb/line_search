import argparse
import numpy as np
import numpy.linalg as la

from functions import Evaluater
from solvers.trust_region import TrustRegion, hebden, cauthy, two_d_subspace_min
from solvers import stable_newton, damp_newton, fletcher_freeman
from step_size import armijo_goldstein_linesearch, wolfe_powell_linesearch

EPSILON = 1e-12
DELTA = 0.1
SAFE_GUARD = 1000

LINE_SEARCH_METHODS = {
    "armijo_goldstein": armijo_goldstein_linesearch,
    "wolfe_powell": wolfe_powell_linesearch,
}

SOLVERS = {
    "sn": stable_newton,
    "dn": damp_newton,
    "ff": fletcher_freeman,
    "hd": TrustRegion(hebden, EPSILON, DELTA),
    "ct": TrustRegion(cauthy, EPSILON, DELTA),
    "tdsm": TrustRegion(two_d_subspace_min, EPSILON, DELTA)
}


def main(func_name, solver_name, ls_method, **kwargs):
    """主函数，用于执行优化算法

    Parameters
    ----------
    func_name : str
        所需优化的函数名
    solver_name : str
        使用的优化算法名
    """
    func = Evaluater(func_name, **kwargs)
    line_search_func = LINE_SEARCH_METHODS[ls_method]
    init = func.init
    solver = SOLVERS[solver_name]
    kwargs.update({
        "line_search_func": line_search_func,
        "safe_guard": SAFE_GUARD
    })
    x, f, g = solver(func, init, **kwargs)

    x_norm = la.norm(x)
    x_mean = x.mean()
    g_norm = la.norm(g)

    if len(x) > 10:
        print("%.4g" % x_norm)
        print("%.4g" % x_mean)
    else:
        print(x)
    print("%.4g" % f)
    print("%.4g" % g_norm)
    print(solver.iters)
    print(func.func_calls)
    print("%.2f" % solver.time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimization Project 3")
    parser.add_argument(
        "-func",
        type=str,
        help="The function you want to optimize",
        default="powell_badly_scaled"
    )
    parser.add_argument(
        "-solver",
        type=str,
        help="The solver that solves this optimization problem"
    )
    parser.add_argument(
        "-ls",
        type=str,
        help="The line search method you want to use",
        default="wolfe_powell"
    )
    parser.add_argument(
        "-m",
        type=int,
        default=8
    )
    args = parser.parse_args()

    print(args)
    np.set_printoptions(precision=4, suppress=True)  # 设置浮点精度

    main(args.func, args.solver, args.ls, m=args.m)

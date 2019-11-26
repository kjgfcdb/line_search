import argparse

import numpy as np
from stackprinter import set_excepthook

from functions import Evaluater
from solvers import inexact_newton, l_bfgs, compact_l_bfgs
from step_size import armijo_goldstein_linesearch, wolfe_powell_linesearch, gll_linesearch, simple_linesearch

set_excepthook(style='darkbg2')

LINE_SEARCH_METHODS = {
    "armijo_goldstein": armijo_goldstein_linesearch,
    "wolfe_powell": wolfe_powell_linesearch,
    "gll": gll_linesearch,
    "simple": simple_linesearch
}

SOLVERS = {
    "inexact_newton": inexact_newton,
    "in": inexact_newton,
    "lbfgs": l_bfgs,
    "clbfgs": compact_l_bfgs
}


def main(func_name, solver_name, **kwargs):
    """主函数，用于执行优化算法

    Parameters
    ----------
    func_name : str
        所需优化的函数名
    solver_name : str
        使用的优化算法名
    """
    func = Evaluater(func_name, **kwargs)
    init = func.init
    solver = SOLVERS[solver_name]
    x, f = solver(func, init, **kwargs)
    print("x: {}\tf: {}".format(repr(x), f))
    print("\n迭代次数\t", solver.iters, "次")
    print("函数调用次数\t", func.func_calls, "次")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Optimization experiments")
    # parser.add_argument(
    #     "-func",
    #     type=str,
    #     help="The function you want to optimize",
    #     default="powell_badly_scaled"
    # )
    # parser.add_argument(
    #     "-solver",
    #     type=str,
    #     help="The solver that solves this optimization problem",
    #     default="damp_newton"
    # )
    # parser.add_argument(
    #     "-n",
    #     type=int,
    #     default=-1,
    # )
    # parser.add_argument(
    #     "-eps",
    #     type=float,
    #     default=1e-8
    # )
    # parser.add_argument(
    #     "-count",
    #     dest='count',
    #     action='store_true'
    # )
    # parser.set_defaults(count=False)
    # args = parser.parse_args()
    # func_name = args.func
    # solver_name = args.solver
    # count = args.count
    # eps = args.eps
    #
    # print(args)
    # np.set_printoptions(precision=4, suppress=True)  # 设置浮点精度
    #
    # main(func_name, solver_name, count=count, eps=eps)
    # main("eps", "in", choice='2', n=1000)
    # main("pi", "clbfgs", n=1000, m=20)

import argparse
import numpy as np

from functions import Evaluater
from solvers.trust_region import TrustRegion, hebden, cauthy
from solvers import stable_newton, damp_newton, fletcher_freeman

EPSILON = 1e-12
DELTA = 0.1
SOLVERS = {
    "sn": stable_newton,
    "dn": damp_newton,
    "ff": fletcher_freeman,
    "hd": TrustRegion(hebden, EPSILON, DELTA),
    "ct": TrustRegion(cauthy, EPSILON, DELTA),
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
    x, f, g = solver(func, init, **kwargs)
    print("x: {}\tf: {}".format(repr(x), f))
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
    #     help="The solver that solves this optimization problem"
    # )
    # parser.add_argument(
    #     "-m",
    #     type=int,
    #     default=4,
    # )
    # args = parser.parse_args()
    # func_name = args.func
    # solver_name = args.solver
    # m = args.m
    #
    # print(args)
    # np.set_printoptions(precision=4, suppress=True)  # 设置浮点精度
    #
    # main("tri", "lbfgs", n=5000, m=9)
    main("eps", "ct", n=20)

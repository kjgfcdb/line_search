import os
import numpy as np
import dill
from sympy import exp, symarray, sqrt, diff, cos, sin
from sympy.utilities.lambdify import lambdify


def fgG(f, x_sympy, **kwargs):
    """对给定的函数进行求导，并返回函数、导数以及Hessian矩阵对应的数值函数

    Parameters
    ----------
    f : sympy表达式
        原始定义的sympy函数
    x_sympy : sympy符号数组
        f的变量

    Returns
    -------
    返回f, g, G，分别是函数、函数的导数、函数的Hessian矩阵对应的数值函数。
    """
#     dill.settings['recurse'] = True
#     file_name = os.path.join("cache", kwargs['func_name'] + "_" + str(kwargs['n']))
#     parent, _ = os.path.split(file_name)
#     if not os.path.exists(parent):
#         os.makedirs(parent)
#     if os.path.isfile(file_name):
#         print("=> 已加载缓存的函数 : " + file_name)
#         with open(file_name, "rb") as fp:
#             func_list = dill.load(fp)
#         return func_list

    use_G = kwargs['use_G'] if 'use_G' in kwargs else True
    f_sympy = f(x_sympy)
    g_sympy = diff(f_sympy, x_sympy).doit()
    if use_G:
        G_sympy = diff(g_sympy, x_sympy).doit()
        func_list = list(lambdify([x_sympy], func, 'numpy')
                         for func in (f_sympy, g_sympy, G_sympy))
    else:
        func_list = list(lambdify([x_sympy], func, 'numpy')
                         for func in (f_sympy, g_sympy))
#     with open(file_name, "wb") as fp:
#         dill.dump(func_list, fp)
#         print("=> 缓存函数 : " + file_name)
    return func_list


def call_counter(count=True):
    """对调用的函数统计其调用次数

    Parameters
    ----------
    count : bool, optional
        是否输出调用次数结果, by default True
    """

    def _call_counter(func):
        num = 0

        def wrapper(*args, **kwargs):
            nonlocal num
            num += 1
            if count:
                print(f"===== call {num} =====")
            return func(*args, **kwargs)

        return wrapper

    return _call_counter


class Evaluater:
    def __init__(self, func_name, **kwargs):
        """代值计算函数，给定函数名，返回一个可以代入求值的具体函数

        Parameters
        ----------
        func_name : str
            函数名
        """
        kwargs['func_name'] = func_name
        n = kwargs['n']
        if func_name == 'extended_powell_singular':
            func_list = extended_powell_singular_numpy(**kwargs)
            init = []
            for i in range(n):
                if i % 4 == 0:
                    init.append(3)
                elif i % 4 == 1:
                    init.append(-1)
                elif i % 4 == 2:
                    init.append(0)
                else:
                    init.append(1)
        elif func_name == "penalty_i":
            func_list = penalty_i_numpy(**kwargs)
            init = [i + 1 for i in range(n)]
        elif func_name == "trigonometric":
            func_list = trigonometric_numpy(**kwargs)
            init = list(1 / n for i in range(n))
        elif func_name == "extended_rosenbrock":
            func_list = extended_rosenbrock_numpy(**kwargs)
            init = []
            for i in range(n // 2):
                init.append(-1.2)
                init.append(1)
        else:
            raise NotImplementedError()
        use_G = kwargs['use_G'] if 'use_G' in kwargs else True
        if use_G:
            f, g, G = func_list
            self.G = G
        else:
            f, g = func_list
            self.G = None
        self.f = f
        self.g = g
        self.init = init
        count = kwargs['count'] if 'count' in kwargs else False
        self.work = call_counter(count)(self.work)

    def work(self, x, g_only):
        """根据输入变量返回得到函数值、导数值、以及Hessian矩阵

        Parameters
        ----------
        x : 数字或者向量
            函数的输入

        Returns
        -------
        返回f,g,G，分别是函数值、导数、Hessian矩阵
        """
        if g_only:
            return np.array(self.g(x))
        f, g = (np.array(func(x)) for func in (self.f, self.g))
        if self.G is not None:
            return f, g, np.array(self.G(x))
        else:
            return f, g

    def __call__(self, x, g_only=False):
        """
        调用work函数
        """
        x = np.array(x)
        return self.work(x, g_only)


def extended_powell_singular_numpy(**kwargs):
    """定义的Extended Powell Singular函数，m是其函数定义中的m
    """
    m = kwargs['n']

    def extended_powell_singular(x):
        r = []
        for i in range(m):
            if i % 4 == 0:  # 1
                temp = x[i] + 10 * x[i + 1]
            elif i % 4 == 1:  # 2
                temp = sqrt(5) * (x[i + 1] - x[i + 2])
            elif i % 4 == 2:  # 3
                temp = (x[i - 1] - 2 * x[i]) ** 2
            else:
                # elif i % 4 == 3:  # 0
                temp = sqrt(10) * (x[i - 3] - x[i]) ** 2
            r.append(temp)
        return sum(item ** 2 for item in r)

    x_sympy = symarray('x', m)
    return fgG(extended_powell_singular, x_sympy, **kwargs)


def penalty_i_numpy(**kwargs):
    n = kwargs['n']
    gamma = 1e-5

    def f(x):
        ret = sum((x-1)**2)*gamma
        ret += sum((n*x**2 - 0.25))**2
        return ret

    def g(x):
        return (2*gamma - n**2 + 4*n**2 * sum(x**2))*x - 2*gamma

    def G(x):
        ret = 8*n**2 * np.outer(x, x)
        ret = ret.astype('float64')
        ret += (4*n**2*x.dot(x)+2*gamma-n**2) * np.eye(n)
        return ret
    return f, g, G


def trigonometric_numpy(**kwargs):
    # n = m
    m = kwargs['n']

    def trigonometric(x):
        r = []
        sum_cos = sum(cos(x[j]) for j in range(m))
        for i in range(m):
            temp = m - sum_cos + (i+1) * (1 - cos(x[i])) - sin(x[i])
            r.append(temp)
        return sum(item ** 2 for item in r)

    x_sympy = symarray('x', m)
    return fgG(trigonometric, x_sympy, **kwargs)


def trigonometric_numpy_new(**kwargs):
    n = kwargs['n']
    def f(x):
        ii = np.arange(1, n+1)
        return sum((n - sum(np.cos(x)) + ii - ii*np.cos(x) - np.sin(x))**2)

    def g(x):
        sum(np.cos(x))
        pass

    def G(x):
        pass
    return f, g, G


def extended_rosenbrock_numpy(**kwargs):
    # n = m
    m = kwargs['n']
    assert (m & 1) == 0

    def extended_rosenbrock(x):
        r = []
        idxs = list(range(m))
        for i, j in zip(idxs[0::2], idxs[1::2]):
            r.append(10 * (x[i + 1] - x[i] ** 2))
            r.append(1 - x[j - 1])
        return sum(item ** 2 for item in r)

    x_sympy = symarray('x', m)
    return fgG(extended_rosenbrock, x_sympy, **kwargs)


if __name__ == "__main__":
    n = 10
    f,g,G = trigonometric_numpy(n = n, func_name = 'foo')
    # f1, g1, G1 = penalty_i_numpy(n=n, func_name='foo')
    # f2, g2, G2 = penalty_i_numpy_new(n=n)
    # for i in range(10):
    #     x = np.random.randn(n)
    #     for a, b in zip((f1, g1, G1), (f2, g2, G2)):
    #         if not np.allclose(a(x), b(x)):
    #             print("fuck")

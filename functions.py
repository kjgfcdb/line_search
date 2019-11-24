import numpy as np
from sympy import diff
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


class Phi_func:
    def __init__(self, func, x0, d, use_G=False):
        """根据f,x0,d构造函数phi
            phi(alpha) = f(x0 + alpha * d)

        Parameters
        ----------
        func : 目标函数
        x0 : 初始点
        d : 下降方向
        use_G : bool, optional
            是否需要phi函数返回Hessian矩阵
        """
        self.func = func
        self.x0 = x0
        self.d = d
        self.use_G = use_G

    def __call__(self, alpha):
        """给定alpha，返回对应的函数值、导数以及Hessian矩阵

        Parameters
        ----------
        alpha : float
            phi函数中的alpha

        Returns
        -------
        返回函数值、导数以及对应的Hessian矩阵
        """
        d = self.d
        if not self.use_G:
            f, g, *_ = self.func(self.x0 + alpha * d)
            return f, g.dot(d)
        else:
            f, g, G = self.func(self.x0 + alpha * d)
            return f, g.dot(d), G


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
        if func_name in ['extended_powell_singular', "eps"]:
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
        elif func_name in ["penalty_i", "pi"]:
            func_list = penalty_i_numpy(**kwargs)
            init = [i + 1 for i in range(n)]
        elif func_name in ["trigonometric", "tri"]:
            func_list = trigonometric_numpy(**kwargs)
            init = list(1 / n for i in range(n))
        elif func_name in ["extended_rosenbrock", "er"]:
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
        x = np.array(x).astype('float')
        return self.work(x, g_only)


def extended_powell_singular_numpy(**kwargs):
    n = kwargs['n']

    def f(x):
        ret = 0
        for i in range(n):
            if i % 4 == 0:
                ret += (x[i] + 10 * x[i + 1]) ** 2
            elif i % 4 == 1:
                ret += 5 * (x[i + 1] - x[i + 2]) ** 2
            elif i % 4 == 2:
                ret += (x[i - 1] - 2 * x[i]) ** 4
            else:
                ret += 10 * (x[i - 3] - x[i]) ** 4
        return ret

    def g(x):
        ret = np.zeros(n)
        for i in range(n):
            if i % 4 == 0:
                ret[i] = 2 * x[i] + 20 * x[i + 1] + 40 * (x[i] - x[i + 3]) ** 3
            elif i % 4 == 1:
                ret[i] = 20 * x[i - 1] + 200 * x[i] + 4 * (x[i] - 2 * x[i + 1]) ** 3
            elif i % 4 == 2:
                ret[i] = 10 * x[i] - 10 * x[i + 1] - 8 * (x[i - 1] - 2 * x[i]) ** 3
            else:
                ret[i] = -10 * x[i - 1] + 10 * x[i] - 40 * (x[i - 3] - x[i]) ** 3
        return ret

    def G(x):
        ret = np.zeros((n, n))
        for i in range(0, n, 4):
            ret[i][i] = 120 * (x[i] - x[i + 3]) ** 2 + 2
            ret[i][i + 1] = 20
            ret[i][i + 2] = 0
            ret[i][i + 3] = -120 * (x[i] - x[i + 3]) ** 2

            ret[i + 1][i] = 20
            ret[i + 1][i + 1] = 12 * (x[i + 1] - 2 * x[i + 2]) ** 2 + 200
            ret[i + 1][i + 2] = -24 * (x[i + 1] - 2 * x[i + 2]) ** 2
            ret[i + 1][i + 3] = 0

            ret[i + 2][i] = 0
            ret[i + 2][i + 1] = -24 * (x[i + 1] - 2 * x[i + 2]) ** 2
            ret[i + 2][i + 2] = 48 * (x[i + 1] - 2 * x[i + 2]) ** 2 + 10
            ret[i + 2][i + 3] = -10

            ret[i + 3][i] = -120 * (x[i] - x[i + 3]) ** 2
            ret[i + 3][i + 1] = 0
            ret[i + 3][i + 2] = -10
            ret[i + 3][i + 3] = 120 * (x[i] - x[i + 3]) ** 2 + 10
        return ret

    return f, g, G


def penalty_i_numpy(**kwargs):
    n = kwargs['n']
    gamma = 1e-5

    def f(x):
        ret = sum((x - 1) ** 2) * gamma
        ret += sum((n * x ** 2 - 0.25)) ** 2
        return ret

    def g(x):
        return (2 * gamma - n ** 2 + 4 * n ** 2 * sum(x ** 2)) * x - 2 * gamma

    def G(x):
        ret = 8 * n ** 2 * np.outer(x, x)
        ret += (4 * n ** 2 * x.dot(x) + 2 * gamma - n ** 2) * np.eye(n)
        return ret

    return f, g, G


def trigonometric_numpy(**kwargs):
    n = kwargs['n']

    def f(x):
        ii = np.arange(1, n + 1)
        return sum((n - sum(np.cos(x)) + ii - ii * np.cos(x) - np.sin(x)) ** 2)

    def g(x):
        ii = np.arange(1, n + 1)
        rhs = ii * (1 - np.cos(x)) + n - np.sin(x) - sum(np.cos(x))
        lhs = np.tile(2 * np.sin(x), (n, 1)).T
        lhs = lhs + np.diag(2 * ii * np.sin(x) - 2 * np.cos(x))
        return lhs.dot(rhs)

    def G(x):
        ii = np.arange(1, n + 1)
        lhs1 = np.tile(2 * np.sin(x), (n, 1)).T
        lhs1 = lhs1 + np.diag(2 * ii * np.sin(x) - 2 * np.cos(x))
        rhs1 = np.tile(np.sin(x), (n, 1)) + np.diag(ii * np.sin(x) - np.cos(x))

        lhs2 = np.tile(2 * np.cos(x), (n, 1)).T
        lhs2 = lhs2 + np.diag(2 * ii * np.cos(x) + 2 * np.sin(x))
        rhs2 = ii * (1 - np.cos(x)) + n - np.sin(x) - sum(np.cos(x))
        res = lhs2.dot(rhs2)
        return lhs1.dot(rhs1) + np.diag(res)

    return f, g, G


def extended_rosenbrock_numpy(**kwargs):
    n = kwargs['n']

    def f(x):
        ret = 0
        idxs = list(range(n))
        for i in idxs[::2]:
            ret += (10 * (x[i + 1] - x[i] ** 2)) ** 2
            ret += (1 - x[i]) ** 2
        return ret

    def g(x):
        idxs = list(range(n))
        ret = np.zeros(n)
        for i in idxs[::2]:
            ret[i] = 400 * x[i] * (x[i] ** 2 - x[i + 1]) + 2 * x[i] - 2
            ret[i + 1] = -200 * x[i] ** 2 + 200 * x[i + 1]
        return ret

    def G(x):
        idxs = list(range(n))
        ret = np.zeros((n, n))
        for i in idxs[::2]:
            ret[i][i] = 1200 * x[i] ** 2 - 400 * x[i + 1] + 2
            ret[i][i + 1] = -400 * x[i]
            ret[i + 1][i] = -400 * x[i]
            ret[i + 1][i + 1] = 200
        return ret

    return f, g, G

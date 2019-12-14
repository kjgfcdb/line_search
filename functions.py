import numpy as np
from sympy import diff, symarray, exp
from sympy.utilities.lambdify import lambdify


def fgG(f, x_sympy):
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
    f_sympy = f(x_sympy)
    g_sympy = diff(f_sympy, x_sympy).doit()
    G_sympy = diff(g_sympy, x_sympy).doit()
    func_list = list(lambdify([x_sympy], func, 'numpy')
                     for func in (f_sympy, g_sympy, G_sympy))
    return func_list


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
        f, g = self.func(self.x0 + alpha * d, use_G=False)
        return f, g.dot(d)


class Evaluater:
    def __init__(self, func_name, **kwargs):
        """代值计算函数，给定函数名，返回一个可以代入求值的具体函数

        Parameters
        ----------
        func_name : str
            函数名
        """
        kwargs['func_name'] = func_name
        if func_name in ['extended_powell_singular', "eps"]:
            m = kwargs['m']
            func_list = extended_powell_singular_numpy(**kwargs)
            init = []
            for i in range(m):
                if i % 4 == 0:
                    init.append(3)
                elif i % 4 == 1:
                    init.append(-1)
                elif i % 4 == 2:
                    init.append(0)
                else:
                    init.append(1)
        elif func_name in ["powell_badly_scaled_numpy", "pbs"]:
            func_list = powell_badly_scaled_numpy()
            init = [0, 1]
        elif func_name in ["biggsexp6", "be6"]:
            func_list = biggs_exp6_numpy(**kwargs)
            init = [1, 2, 1, 1, 1, 1]
        else:
            raise NotImplementedError()

        self.f, self.g, self.G = func_list

        self.init = init
        self.func_calls = 0

    def __call__(self, x, g_only=False, use_G=True):
        """根据输入变量返回得到函数值、导数值、以及Hessian矩阵

        Parameters
        ----------
        x : 数字或者向量
            函数的输入

        Returns
        -------
        返回f,g,G，分别是函数值、导数、Hessian矩阵
        """
        self.func_calls += 1
        x = np.array(x).astype('float')
        if g_only:
            return np.array(self.g(x))
        if not use_G:
            f, g = (np.array(func(x)) for func in (self.f, self.g))
            return f, g
        f, g, G = (np.array(func(x)) for func in (self.f, self.g, self.G))
        return f, g, G


def extended_powell_singular_numpy(**kwargs):
    """定义的extended powell singular函数
    """
    m = kwargs['m']

    def f(x):
        ret = 0
        for i in range(m):
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
        ret = np.zeros(m)
        for i in range(m):
            if i % 4 == 0:
                ret[i] = 2 * x[i] + 20 * x[i + 1] + 40 * (x[i] - x[i + 3]) ** 3
            elif i % 4 == 1:
                ret[i] = 20 * x[i - 1] + 200 * x[i] + \
                         4 * (x[i] - 2 * x[i + 1]) ** 3
            elif i % 4 == 2:
                ret[i] = 10 * x[i] - 10 * x[i + 1] - \
                         8 * (x[i - 1] - 2 * x[i]) ** 3
            else:
                ret[i] = -10 * x[i - 1] + 10 * \
                         x[i] - 40 * (x[i - 3] - x[i]) ** 3
        return ret

    def G(x):
        ret = np.zeros((m, m))
        for i in range(0, m, 4):
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


def powell_badly_scaled_numpy():
    """定义的Powell Badly Scaled函数
    """

    def powell_badly_scaled(x):
        x1 = x[0]
        x2 = x[1]
        r1 = 1e4 * x1 * x2 - 1
        r2 = exp(-x1) + exp(-x2) - 1.0001
        return r1 ** 2 + r2 ** 2

    x_sympy = symarray('x', 2)
    return fgG(powell_badly_scaled, x_sympy)


def biggs_exp6_numpy(**kwargs):
    """定义的Biggs EXP6函数，m是其函数定义中的m
    """
    #  m = 8,9,10,11,12
    m = kwargs['m']

    def biggs_exp6(x):
        r = []
        for i in range(m):
            ti = 0.1 * (i + 1)
            yi = exp(-ti) - 5 * exp(-10 * ti) + 3 * exp(-4 * ti)
            temp = x[2] * exp(-ti * x[0]) - x[3] * \
                   exp(-ti * x[1]) + x[5] * exp(-ti * x[4]) - yi
            r.append(temp)
        return sum(item ** 2 for item in r)

    x_sympy = symarray('x', 6)
    return fgG(biggs_exp6, x_sympy)


class Q_func:
    def __init__(self, f, g, G):
        self.f = f
        self.g = g
        self.G = G

    def __call__(self, d):
        return self.f + self.g.dot(d) + 0.5 * d.dot(self.G).dot(d)

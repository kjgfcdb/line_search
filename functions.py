import numpy as np
from sympy import exp, symarray, sqrt, diff
from sympy.utilities.lambdify import lambdify


def fgG(f, x_sympy):
    f_sympy = f(x_sympy)
    g_sympy = diff(f_sympy, x_sympy)
    G_sympy = diff(g_sympy, x_sympy)
    f_numpy, g_numpy, G_numpy = cvt2numpy(f_sympy, g_sympy, G_sympy, x_sympy)
    return f_numpy, g_numpy, G_numpy


def cvt2numpy(f_sympy, g_sympy, G_sympy, x_sympy):
    f_numpy = lambdify([x_sympy], f_sympy, 'numpy')
    g_numpy = lambdify([x_sympy], g_sympy.doit(), 'numpy')
    G_numpy = lambdify([x_sympy], G_sympy.doit(), 'numpy')
    return f_numpy, g_numpy, G_numpy


class Evaluater:
    def __init__(self, func_name, **kwargs):
        if func_name == "powell_badly_scaled":
            f, g, G = powell_badly_scaled_numpy()
        elif func_name == "biggs_exp6":
            m = kwargs['m']
            f, g, G = biggs_exp6_numpy(m)
        elif func_name == 'extended_powell_singular':
            m = kwargs['m']
            f, g, G = extended_powell_singular_numpy(m)
        else:
            raise NotImplementedError()
        self.f = f
        self.g = g
        self.G = G

    def __call__(self, x):
        f, g, G = (np.array(func(x)) for func in (self.f, self.g, self.G))
        return f, g, G


class Phi_func:
    def __init__(self, func, x0, d, use_G=False):
        self.func = func
        self.x0 = x0
        self.d = d
        self.use_G = use_G

    def __call__(self, alpha):
        d = self.d
        if not self.use_G:
            f, g, *_ = self.func(self.x0 + alpha * d)
            return f, g.dot(d)
        else:
            f, g, G = self.func(self.x0 + alpha * d)
            return f, g.dot(d), G


def powell_badly_scaled_numpy():
    def powell_badly_scaled(x):
        x1 = x[0]
        x2 = x[1]
        r1 = 1e4 * x1 * x2 - 1
        r2 = exp(-x1) + exp(-x2) - 1.0001
        return r1 ** 2 + r2 ** 2

    x_sympy = symarray('x', 2)
    return fgG(powell_badly_scaled, x_sympy)


def biggs_exp6_numpy(m):
    def biggs_exp6(x):
        r = []
        for i in range(m):
            ti = 0.1 * (i + 1)
            yi = exp(-ti) - 5 * exp(-10 * ti) + 3 * exp(-4 * ti)
            temp = x[2] * exp(-ti * x[0]) - x[3] * exp(-ti * x[1]) + x[5] * exp(-ti * x[4]) - yi
            r.append(temp)
        return sum(item ** 2 for item in r)

    x_sympy = symarray('x', 6)
    return fgG(biggs_exp6, x_sympy)


def extended_powell_singular_numpy(m):
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
    return fgG(extended_powell_singular, x_sympy)

import numpy as np
from sympy import exp, symarray, sqrt, diff
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
    g_sympy = diff(f_sympy, x_sympy)
    G_sympy = diff(g_sympy, x_sympy)
    f_numpy, g_numpy, G_numpy = cvt2numpy(f_sympy, g_sympy, G_sympy, x_sympy)
    return f_numpy, g_numpy, G_numpy


def cvt2numpy(f_sympy, g_sympy, G_sympy, x_sympy):
    """将sympy类型的函数转换为numpy类的函数，使之可以进行数值计算
    
    Parameters
    ----------
    f_sympy : sympy表达式
        sympy类型的函数定义
    g_sympy : sympy表达式
        f的导数函数 
    G_sympy : sympy表达式
        f的Hessian函数
    x_sympy : sympy符号数组
        f的输入变量
    
    Returns
    -------
    返回f_numpy, g_numpy, G_numpy，分别是函数、函数的导数、函数的Hessian矩阵对应的数值函数
    """
    f_numpy = lambdify([x_sympy], f_sympy, 'numpy')
    g_numpy = lambdify([x_sympy], g_sympy.doit(), 'numpy')
    G_numpy = lambdify([x_sympy], G_sympy.doit(), 'numpy')
    return f_numpy, g_numpy, G_numpy


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
        if func_name == "powell_badly_scaled":
            f, g, G = powell_badly_scaled_numpy()
            m = 2
            init = [0, 1]
        elif func_name == "biggs_exp6":
            m = kwargs['m'] if kwargs['m'] > 0 else 8
            f, g, G = biggs_exp6_numpy(m)
            init = [1, 2, 1, 1, 1, 1]
        elif func_name == 'extended_powell_singular':
            m = kwargs['m'] if kwargs['m'] > 0 else 20
            f, g, G = extended_powell_singular_numpy(m)
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
        else:
            raise NotImplementedError()
        self.f = f
        self.g = g
        self.G = G
        self.init = init
        count = kwargs['count']
        self.work = call_counter(count)(self.work)

    def work(self, x):
        """根据输入变量返回得到函数值、导数值、以及Hessian矩阵
        
        Parameters
        ----------
        x : 数字或者向量
            函数的输入
        
        Returns
        -------
        返回f,g,G，分别是函数值、导数、Hessian矩阵
        """
        f, g, G = (np.array(func(x)) for func in (self.f, self.g, self.G))
        return f, g, G

    def __call__(self, x):
        """
        调用work函数
        """
        f, g, G = self.work(x)
        return f, g, G


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


def biggs_exp6_numpy(m):
    """定义的Biggs EXP6函数，m是其函数定义中的m
    """
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


def extended_powell_singular_numpy(m):
    """定义的Extended Powell Singular函数，m是其函数定义中的m
    """
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

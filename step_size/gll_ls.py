from collections import deque
from conditions import gll

class gll_linesearch:
    def __init__(self, **kwargs):
        """GLL线搜索准则
        """
        queue_size = kwargs['queue_size'] if 'queue_size' in kwargs else 10
        self.q = deque(maxlen=queue_size)

    def __call__(self, phi, stepsize=1, rho=0.5, sigma=0.5, **kwargs):
        """调用GLL线搜索准则求得步长
        
        Parameters
        ----------
        phi : 函数
            定义的phi函数，phi(alpha) = f(x0 + alpha * d)
        stepsize : float
            初始步长
        rho : float
            GLL准则中的\rho
        sigma : float
            GLL准则中的\sigma
        
        Returns
        -------
        返回搜索得到的步长
        """
        f, g = phi(0)
        self.q.append(f)
        safe_guard = kwargs['safe_guard'] if 'safe_guard' in kwargs else None
        k = 0
        while True:
            f_new, _ = phi(stepsize)
            if safe_guard is not None and k > safe_guard:
                break
            if not gll(f_new, max(self.q), stepsize, g, None, rho):
                stepsize = sigma * stepsize
            else:
                break
            k += 1
        return stepsize

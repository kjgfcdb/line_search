from scipy.linalg import ldl


def bunch_parlett(A):
    """BP分解
    
    Parameters
    ----------
    A : 输入的矩阵
    
    Returns
    -------
    返回BP分解后的L、D矩阵
    """
    # FIXME: this is bunch-kaufman, not bunch_parlett
    L, D, perm = ldl(A)
    L = L[perm, :]
    return L, D

from scipy.linalg import ldl


def bunch_parlett(A):
    # FIXME: this is bunch-kaufman, not bunch_parlett
    L, D, perm = ldl(A)
    L = L[perm, :]
    return L, D

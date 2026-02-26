from numba import jit
import numpy as np


@jit(nopython=True,cache=True)
def DD_table(y:np.ndarray,dx:float,order:int) -> np.ndarray:
    n = len(y)
    levels = order + 1
    out = np.zeros((n, levels), dtype=float)

    out[:, 0] = y
    for level in range(1, levels):
        for i in range(n - level):
            out[i, level] = (out[i+1, level-1] - out[i, level-1]) / (level * dx)

    return out


@jit(nopython=True,cache=True)
def construct_interpolating_poly():
    pass


@jit(nopython=True,cache=True)
def functional_newton(max_iter=4):
    raise RuntimeError
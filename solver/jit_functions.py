from numba import jit
import numpy as np
import sympy as sp


@jit(nopython=True,cache=True)
def newton_coeffs(x:np.ndarray,y:np.ndarray,order:int) -> np.ndarray:
    n = len(y)
    levels = order
    dd = np.zeros((n, levels), dtype=float)

    dd[:, 0] = y
    for level in range(1, levels):
        for i in range(n - level):
            dd[i, level] = (dd[i+1, level-1] - dd[i, level-1]) / (x[i+level] - x[i])

    return dd[0, :]


@jit(nopython=True,cache=True)
def interpolate(x_data:np.ndarray,y_data:np.ndarray,x_query:np.ndarray) -> np.ndarray:
    n = len(x_data)
    coeffs = newton_coeffs(x_data,y_data,order=n)

    out = coeffs[0]
    mult_x = 1.0

    for k in range(1, n):
        mult_x *= (x_query - x_data[k - 1])
        out += coeffs[k] * mult_x

    return out


@jit(nopython=True,cache=True)
def interpolate_sym(xsym:sp.Symbol,x_data:np.ndarray,y_data:np.ndarray) -> np.ndarray:
    n = len(x_data)
    coeffs = newton_coeffs(x_data,y_data,order=n)

    poly = coeffs[0]
    mult_x = 1.0

    for k in range(1, n):
        mult_x *= (xsym - x_data[k - 1])
        poly += coeffs[k] * mult_x

    return poly

@jit(nopython=True,cache=True)
def functional_newton_iteration(max_iter=4):
    raise RuntimeError
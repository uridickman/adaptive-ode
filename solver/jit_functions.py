from numba import jit
import numpy as np


@jit(nopython=True,cache=True)
def newton_coeffs(
        x           :np.ndarray,
        y           :np.ndarray,
        order       :int
    )               -> np.ndarray:
    n = len(y)
    levels = order
    dd = np.zeros((n, levels), dtype=float)

    dd[:, 0] = y
    for level in range(1, levels):
        for i in range(n - level):
            dd[i, level] = (dd[i+1, level-1] - dd[i, level-1]) / (x[i+level] - x[i])

    return dd[0, :]


@jit(nopython=True,cache=True)
def interpolate(
        x_data      :np.ndarray,
        y_data      :np.ndarray,
        x_query     :np.ndarray,
    )               -> np.ndarray:

    n = len(x_data)
    coeffs = newton_coeffs(x_data,y_data,order=n)

    out    = np.zeros(len(x_query), dtype=np.float64)
    mult_x = np.ones(len(x_query),  dtype=np.float64)
    out[:] = coeffs[0]

    for k in range(1, n):
        mult_x = mult_x * (x_query - x_data[k - 1])
        out = out + coeffs[k] * mult_x

    return out


@jit(nopython=True, cache=True)
def adams_order_1(ynm1, f, hn):
    return ynm1 + f*hn

@jit(nopython=True, cache=True)
def adams_moulton(ynm1, fn, fnm1, hn):
    return ynm1 + hn/2*(fn+fnm1)

@jit(nopython=True, cache=True)
def adams_bashforth_2(ynm1, fnm1, fnm2, hn,hnm1):
    return ynm1 + fnm1*hn + (fnm1-fnm2)*2*hn*hn/hnm1

@jit(nopython=True, cache=True)
def adams_bashforth_1(ynm1, fnm1, hn):
    return ynm1 + fnm1*hn   
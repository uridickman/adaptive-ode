from numba import jit
import numpy as np
import sympy as sp


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
def integral_of_interpolant_2o(xquery, x_data, y_data, order):
    assert order <= 2 and order > 0, "Order cannot exceed 2 or preceed 0."
    coeffs = newton_coeffs(x_data, y_data, order=order)
    match order:
        case 0:
            return coeffs[0] * xquery

        case 1:
            x0 = x_data[0]
            return (
                coeffs[0] * xquery
                + coeffs[1] * (xquery**2 / 2 - x0 * xquery)
            )

        case 2:
            x0 = x_data[0]
            x1 = x_data[1]

            return (
                coeffs[0] * xquery
                + coeffs[1] * (xquery**2 / 2 - x0 * xquery)
                + coeffs[2] * (xquery**3 / 3 - (x0 + x1) * xquery**2 / 2 + x0 * x1 * xquery
                )
            )
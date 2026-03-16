from numba import njit,types
from numba.experimental import jitclass
from typing import Callable
import numpy as np

spec = [
    ('t', types.float64),
    ('h', types.float64),
    ('f', types.float64[:]),
    ('y', types.float64[:]),
]

@jitclass(spec)
class StepState:
    def __init__(self, t, h, f, y):
        self.t = t
        self.h = h
        self.f = f
        self.y = y

@njit(cache=True)
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


@njit(cache=True)
def interpolate(
        x_data      :np.ndarray,
        y_data      :np.ndarray,
        x_query     :np.ndarray,
    )               -> np.ndarray:

    n = len(x_data)
    coeffs = newton_coeffs(x_data,y_data,n)

    out    = np.zeros(len(x_query), dtype=np.float64)
    mult_x = np.ones(len(x_query),  dtype=np.float64)
    out[:] = coeffs[0]

    for k in range(1, n):
        mult_x = mult_x * (x_query - x_data[k - 1])
        out = out + coeffs[k] * mult_x

    return out

@njit(cache=True)
def lte(ypred:np.ndarray,ycorrect:np.ndarray):
    return np.linalg.norm(ypred-ycorrect,ord=2)

@njit(cache=True)
def new_step_size(est,hn,frac,etol,p):
    return (frac * etol / est)**(1/(p+1))*hn

@njit
def unpack_step_states(step_states: types.List):
    n = len(step_states)
    m = step_states[0].y.shape[0]

    tvec = np.empty(n, dtype=np.float64)
    hvec = np.empty(n-1, dtype=np.float64)
    yvec = np.empty((n,m), dtype=np.float64)
    fvec = np.empty((n,m), dtype=np.float64)

    for i in range(n):
        tvec[i] = step_states[i].t
        yvec[i] = step_states[i].y
        fvec[i] = step_states[i].f
        if i > 0:
            hvec[i-1] = step_states[i].h

    return tvec, yvec, fvec, hvec

@njit
def correct(
        t:np.float64,
        hn:np.float64,
        f:types.Callable,
        state_prev:StepState,
        state_predict:StepState
        
    ):
    ynm1 = state_prev.y
    fnm1 = state_prev.f
    y_predict = state_predict.y
    fn = f(t, y_predict)
    y_correct = adams_moulton_2(ynm1, fn, fnm1, hn)
    return StepState(
        t,
        hn,
        f(t,y_correct),
        y_correct
    )

@njit
def predict_first(
        t:np.float64,
        hn:np.float64,
        f:Callable,
        state_prev:StepState
    ):
    y_predict = adams_bashforth_1(state_prev.y, state_prev.f, hn)
    return StepState(
        t,
        hn,
        f(t,y_predict),
        y_predict
    )

@njit
def predict(
        t:np.float64,
        hn:np.float64,
        f:Callable,
        state_prev:StepState,
        state_prev2:StepState
    ):

    y_predict = adams_bashforth_2(state_prev.y, state_prev.f, state_prev2.f, hn, state_prev.h)
    return StepState(
        t,
        hn,
        f(t,y_predict),
        y_predict
    )

@njit
def adams_moulton_2(ynm1, fn, fnm1, hn):
    return ynm1 + hn/2*(fn+fnm1)

@njit
def adams_bashforth_2(ynm1, fnm1, fnm2, hn, hnm1):
    return ynm1 + fnm1*hn + (fnm1-fnm2)*hn*hn/2/hnm1

@njit
def adams_bashforth_1(ynm1, fnm1, hn):
    return ynm1 + fnm1*hn   
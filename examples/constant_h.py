from solver import ODESolver
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from .plot_util import plot

y0 = np.array([1,2])
trange = (0,1)
h = 0.01

@njit(cache=True)
def f(t,y):
    out = np.empty(2, dtype=np.float64)
    out[0] = -y[0]
    out[1] = -10*(y[1]-t*t) + 2*t
    return out

def solve_constant_h():
    solver = ODESolver(
        f=f,
        y0=y0,
        trange=trange,
        h=h
    )
    solver.solve()
    
    sol = (
        np.copy(solver.T),
        np.copy(solver.Y),
        np.copy(solver.H)
    )

    solver.solve()

    sol = (
        np.copy(solver.T),
        np.copy(solver.Y),
        np.copy(solver.H)
    )

    fig, axs = plt.subplots(2,2,figsize=(12,12),constrained_layout=True)
    plot(*sol,axs,None,"red","solid")
    fig.savefig("figs/constant_h.png",dpi=300)
from solver import ODESolver
import numpy as np
from numba import njit
from .plot_util import plot
import matplotlib.pyplot as plt

y0 = np.array([2,0])
trange = (0,11)

@njit(cache=True)
def f(t,y):
    out = np.empty(2, dtype=np.float64)
    out[0] = y[1]
    out[1] = 2*((1-y[0]*y[0])*y[1]-y[0])
    return out

def solve_VanDerPol():
    solver = ODESolver(
        f=f,
        y0=y0,
        trange=trange,
        etol=1e-3
    )
    solver.solve()
    
    sol1 = (
        np.copy(solver.T),
        np.copy(solver.Y),
        np.copy(solver.H)
    )

    solver._etol = 1e-6
    solver.solve()

    sol2 = (
        np.copy(solver.T),
        np.copy(solver.Y),
        np.copy(solver.H)
    )

    fig, axs = plt.subplots(2,2,figsize=(12,12),constrained_layout=True)
    plot(*sol1,axs,f"etol={1e-3}","red","solid")
    plot(*sol2,axs,f"etol={1e-6}","dodgerblue","dashed")
    fig.savefig("figs/vanderpol.png",dpi=300)
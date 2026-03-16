from solver import ODESolver
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp

y0 = np.array([10,10])
trange = (0,100)

@njit(cache=True)
def f(t,y):
    out = np.empty(2, dtype=np.float64)
    out[0] = 0.25*y[0] - 0.01*y[0]*y[1]
    out[1] = -y[1] + 0.01*y[0]*y[1]
    return out

def solve_PredatorPrey(etol):
    solver = ODESolver(
        f=f,
        y0=y0,
        trange=trange,
        etol=etol
    )
    solver.solve()
    
    Y = solver.Y
    T = solver.T
    H = solver.H

    y1 = Y[:,0]
    y2 = Y[:,1]

    sol = solve_ivp(f, trange, y0, method='RK45', dense_output=True, rtol=etol, atol=etol)

    y_eval = sol.sol(T)
    y1_eval,y2_eval = y_eval[0],y_eval[1]

    _, (ax1,ax2) = plt.subplots(1,2)

    ax1.plot(y1, y2,label="My solver",color="red",linewidth=2)
    ax1.plot(y1_eval, y2_eval,label="Scipy",color="dodgerblue",linestyle="dashed",linewidth=2)
    ax1.legend()

    ax1.set_xlabel("y1")
    ax1.set_ylabel("y2")

    ax2.plot(solver.H)
    ax2.set_xlabel("Step number")
    ax2.set_ylabel("Step size (s)")
    ax2.set_yscale("log")

    plt.show()

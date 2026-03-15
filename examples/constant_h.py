from solver import ODESolver
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

y0 = np.array([1,2])
trange = (0,1)
h = 0.01

def f(t,y): return np.array([-y[0],-10*(y[1]-t*t) + 2*t])

def solve_constant_h():
    solver = ODESolver(
        f=f,
        y0=y0,
        trange=trange,
        h=h
    )
    solver.solve()
    
    Y = solver.Y
    T = solver.T
    H = solver.H

    y1 = Y[:,0]
    y2 = Y[:,1]

    sol = solve_ivp(f, trange, y0, method='RK45', dense_output=True, rtol=1e-3, atol=1e-3,vectorized=True)

    y_eval = sol.sol(T)
    y1_eval,y2_eval = y_eval[0],y_eval[1]

    _, (ax1,ax2) = plt.subplots(1,2)

    ax1.plot(y1, y2,label="My solver",color="red",linewidth=2)
    ax1.plot(y1_eval, y2_eval,label="Scipy",color="dodgerblue",linestyle="dashed",linewidth=2)
    ax1.legend()

    ax1.set_xlabel("y1")
    ax1.set_ylabel("y2")

    ax2.plot(H)
    ax2.set_xlabel("Step num")
    ax2.set_ylabel("Step size (s)")
    ax2.set_yscale("log")

    plt.show()
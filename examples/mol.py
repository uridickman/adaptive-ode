from solver import ODESolver
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

trange = (0,1)
xrange = (0,1)
k = 4
dx = (xrange[1] - xrange[0]) / k
X = np.linspace(*xrange,num=k)
y0 = np.exp(-10*X)

diag = -1*np.ones(k+1)
ldiag = np.ones(k)
A  = (np.diag(diag,k=0)+np.diag(ldiag,k=-1)) / dx
A[0,0] = 1 # Enforce boundary condition

t_eval = np.array([0, 0.25 ,0.5, 0.6, 0.8, 1])

def f(t,y): return A @ y

def solve_MoL(etol):
    solver = ODESolver(
        f=f,
        y0=y0,
        trange=trange,
        y_at=t_eval,
        etol=etol
    )
    solver.solve()
    
    Y = solver.Y
    T = solver.T
    H = solver.H

    y1 = Y[:,0]
    y2 = Y[:,-1]

    sol = solve_ivp(f, trange, y0, method='RK45', dense_output=True, rtol=etol, atol=etol,vectorized=True)

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
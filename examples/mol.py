from solver import ODESolver
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp

trange = (0,1)
xrange = (0,1)
k = 100
dx = (xrange[1] - xrange[0]) / k
X = np.linspace(*xrange,num=k+1)
y0 = np.exp(-10*X)

diag = -1*np.ones(k+1)
ldiag = np.ones(k)
A  = (np.diag(diag,k=0)+np.diag(ldiag,k=-1)) / dx
A[0,0] = 1

t_eval = np.array([0, 0.25 ,0.5, 0.6, 0.8, 1])

@njit(cache=True)
def f(t,y): return A @ y

def solve_MoL(etol):
    solver = ODESolver(
        f=f,
        y0=y0,
        trange=trange,
        t_eval=t_eval,
        etol=etol
    )
    solver.solve()
    
    Y = solver.Y
    T = solver.T

    sol = solve_ivp(f, trange, y0, method='RK45', dense_output=True, rtol=etol, atol=etol)

    Y_eval = sol.sol(T).T

    _, ax = plt.subplots()

    for t,y,y_eval in zip(T,Y,Y_eval):
        ax.plot(X, y,label=f"t={t:.2f}")
        ax.plot(X, y_eval,linestyle="dashed",color="black",alpha=0.5,linewidth=2)
    ax.plot([],[],linestyle="dashed",color="grey",linewidth=2,label="Scipy")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y(t,x)")

    plt.show()
from solver import ODESolver
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

fs = 16
ts = 14
lw = 2

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
def f(t,y): return A @ np.ascontiguousarray(y)

def solve_MoL():
    solver = ODESolver(
        f=f,
        y0=y0,
        trange=trange,
        t_eval=t_eval,
        etol=1e-3
    )
    solver.solve()
    
    Y = solver.Y
    T = solver.T

    fig, ax = plt.subplots(figsize=(6,6))

    for t,y in zip(T,Y):
        ax.plot(X, y,label=f"t={t:.2f}",linewidth=lw)

    ax.legend(fontsize=14)
    ax.set_xlabel("x",fontsize=fs)
    ax.set_ylabel("y(t,x)",fontsize=fs)
    ax.tick_params("both",labelsize=ts)

    fig.savefig("figs/mol.png",dpi=300)
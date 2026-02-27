from ..solver import ODESolver
import numpy as np

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

t = np.array([0, 0.25 ,0.5, 0.6, 0.8, 1])

def f(t,y): return A @ y

def solve_MoL(rtol):
    solver = ODESolver(
        f=f,
        y0=y0,
        trange=trange,
        rtol=rtol
    )
    solver.solve()
    
    Y = solver.Y
    T = solver.T

    return T,Y
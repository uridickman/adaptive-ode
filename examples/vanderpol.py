from ..solver import ODESolver
import numpy as np

y0 = np.array([2,0])
trange = (0,11)

def f(t,y):
    return np.array([
        y[1],
        2*((1-y[0]*y[0])*y[1]-y[0])
    ])

def solve_VanDerPol(rtol):
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
from ..solver import ODESolver
import numpy as np

y0 = np.array([10,10])
trange = (0,100)

def f(t,y):
    return np.array([
        0.25*y[0] - 0.01*y[0]*y[1],
        -y[1] + 0.01*y[0]*y[1]
    ])

def solve_PredatorPrey(rtol):
    solver = ODESolver(
        f=f,
        y0=y0,
        trange=trange,
        rtol=rtol
    )
    solver.solve()
    
    Y = solver.Y
    T = solver.T
    H = solver.H

    return T,Y,H

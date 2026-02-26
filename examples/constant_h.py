from ..solver import ODESolver
import numpy as np

y0 = [1,2]
trange = (0,1)
h = 0.01

def f(t,y): return np.array([-y[0],-10*(y[1]-t*t) + 2*t])

def solve_constant_h():
    solver = ODESolver(f=f,y0=y0,trange=trange,h=h)
    solver.solve()
    
    Y = solver.Y
    T = solver.T

    return T,Y
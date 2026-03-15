from ..solver import ODESolver
import numpy as np
import matplotlib.pyplot as plt

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

    return T,Y,H

T,Y,H = solve_constant_h()
y1 = Y[:,0]
y2 = Y[:,1]

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(T, y1, label="y1")
ax1.plot(T, y2, label="y2")

ax1.set_xlabel("t")
ax1.set_ylabel("state")

ax2.plot(T,H)

plt.show()
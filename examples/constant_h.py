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

    return T,Y

T,Y = solve_constant_h()
y1 = Y[:,0]
y2 = Y[:,1]

fig, ax = plt.subplots()

ax.plot(T, y1, label="y1")
ax.plot(T, y2, label="y2")

ax.set_xlabel("t")
ax.set_ylabel("state")

ax.legend()
ax.grid(True)

plt.show()
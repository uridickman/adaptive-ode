import numpy as np
from jit_functions import *
from typing import Tuple,Callable
from warnings import warn

class ODESolver:
    def __init__(
            self,
            f           : Callable,
            y0          : np.ndarray,
            trange      : Tuple[float,float],
            y_at        : np.ndarray=None,
            rtol        : float=1e-3,
            h           : float=-1
        ):
        
        self.tmin,self.tmax = trange
        self.Y0 = y0

        self.use_adaptive_h = h < 0
        self.h = 0.1 if self.use_adaptive_h else h
        self.rtol = rtol
        self.y_at = np.sort(y_at)

        self.f = f
        self.H = [self.h]
        self.T = [self.tmin]
        self.solution_list = [self.Y0]

        self._max_iter = 5

    def solve(self):
        # Solve ODE here
        while t < self.tmax:
            t = self.T[-1]
            if self.use_adaptive_h:
                self.h = self.compute_step_size()
                self.H.append(self.h)
            t += self.h
            self.T.append(t)
            
        self.Y = np.array(self.solution_list)
        

    def predict(self):
        pass

    def correct(self,ypred):
        pass

    def adams_moulton_step(self):
        pass

    def adams_bashforth_step(self):
        pass

    def compute_step_size(self):
        num_iter = 0
        while num_iter < self._max_iter:
            pass
        if num_iter >= self._max_iter:
            warn("Step size failed to converge.",category=RuntimeWarning)
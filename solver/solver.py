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
            rtol        : float=1e-3,
            h           : float=-1
        ):
        
        self.tmin,self.tmax = trange
        self.Y0 = y0

        self.use_adaptive_h = h < 0
        self.h = 0.1 if self.use_adaptive_h else h
        self.rtol = rtol

        self.f = f
        self.H = [self.h]
        self.T = [self.tmin]
        self.solution_list = [self.Y0]

        self._max_iter = 5

    def solve(self):
        # Solve ODE here
        t = self.tmin
        while t < self.tmax:
            self.h = self.compute_step_size
            ypred = self.predict()
            ycorrect = self.correct(ypred)
            t += self.h

        self.T.append(self.T[-1] + self.h)

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
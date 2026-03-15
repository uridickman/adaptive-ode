import numpy as np
from .jit_functions import *
from typing import Tuple,Callable
from dataclasses import dataclass
from warnings import warn

@dataclass
class StepState:
    t: float
    h: float
    f: np.ndarray
    y: np.ndarray


class ODESolver:
    def __init__(
            self,
            f           : Callable,
            y0          : np.ndarray,
            trange      : Tuple[float,float],
            y_at        : np.ndarray=None,
            etol        : float=1e-3,
            h           : float=-1
            
        ):
        
        self.tmin,self.tmax = trange
        self.f = f
        self.h = etol
        self.step_states = []

        self._etol = etol
        self._frac = 0.9
        self._p = 2
        self._max_iter = 15
        

        step0 = StepState(
            self.tmin,
            self.h,
            y0,
            self.f(self.tmin,y0)
        )

        self.step_states.append(step0)
        self.t = self.tmin


    def unpack_step_states(self):
        self.T = np.array([s.t for s in self.step_states])
        self.Y = np.array([s.y for s in self.step_states])
        self.F = np.array([s.f for s in self.step_states])
        self.H = np.array([s.h for s in self.step_states])


    def solve(self):
        self.h = self.predict_step_size(first_step=True)
        self.t += self.h
        y1 = self.take_first_step(self.h)
        self.step_states.append(y1)
        while self.t < self.tmax:
            self.h = self.predict_step_size()
            self.t += self.h
            yn = self.take_step(self.h)
            self.step_states.append(yn)
        self.unpack_step_states()


    def lte(self,yn,ynm1):
        return np.linalg.norm(yn-ynm1,ord=2)


    def new_step_size(self,state_n,est):
        return (self._frac * self._etol / est)**(1/(self._p+1))*state_n.h


    def predict_step_size(self, first_step=False):
        ynm1 = self.step_states[-1].y
        h = self.h if first_step else self.step_states[-1].h
        
        for _ in range(self._max_iter):
            state_guess = self.take_first_step(h) if first_step else self.take_step(h)
            est = self.lte(state_guess.y, ynm1)
            if est <= self._etol:
                return h
            h = self.new_step_size(state_guess, est)
    
        raise RuntimeError("Step size did not converge.")

    def _correct(self, ynm1, fnm1, y_predict, hn):
        fn = self.f(self.t, y_predict)
        y_correct = adams_moulton(ynm1, fn, fnm1, hn)
        return StepState(
            t=self.t,
            h=hn,
            y=y_correct,
            f=self.f(self.t, y_correct)
        )

    def take_first_step(self, hn):
        state_prev = self.step_states[-1]
        y_predict = adams_bashforth_1(state_prev.y, state_prev.f, hn)
        return self._correct(state_prev.y, state_prev.f, y_predict, hn)

    def take_step(self, hn):
        state_prev  = self.step_states[-1]
        state_prev2 = self.step_states[-2]
        y_predict = adams_bashforth_2(state_prev.y, state_prev.f, state_prev2.f, hn, state_prev.h)
        return self._correct(state_prev.y, state_prev.f, y_predict, hn)
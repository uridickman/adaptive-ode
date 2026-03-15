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
        self.h = 0.1
        self.step_states = []

        self._etol = etol
        self._frac = 0.9
        self._p = 2
        self._max_iter = 15
        

        step0 = StepState(
            t=self.tmin,
            h=self.h,
            y=y0,
            f=self.f(self.tmin,y0)
        )

        self.step_states.append(step0)
        self.t = float(self.tmin)
        self.h = self.predict_step_size(first_step=True)
        step0.h = self.h


    def unpack_step_states(self):
        self.T = np.array([s.t for s in self.step_states])
        self.Y = np.array([s.y for s in self.step_states])
        self.F = np.array([s.f for s in self.step_states])
        self.H = np.array([s.h for s in self.step_states[1:]])


    def solve(self):
        _,state_next = self.take_first_step(self.h)
        self.step_states.append(state_next)

        while self.t < self.tmax:
            self.h = self.predict_step_size()
            self.t += self.h
            _,state_next = self.take_step(self.h)
            self.step_states.append(state_next)

        self.unpack_step_states()


    def lte(self,ypred,ycorrect):
        return np.linalg.norm(ypred-ycorrect,ord=2)


    def new_step_size(self,state_n,est):
        return (self._frac * self._etol / est)**(1/(self._p+1))*state_n.h


    def predict_step_size(self,first_step=False):
        h = self.h
        
        for _ in range(self._max_iter):
            state_predict,state_correct = self.take_first_step(h) if first_step else self.take_step(h)
            est = self.lte(state_predict.y,state_correct.y)
            if est <= self._etol:
                return h
            h = self.new_step_size(state_correct, est)
    
        raise RuntimeError("Step size did not converge.")

    def _correct(self, state_prev, state_predict, hn):
        ynm1 = state_prev.y
        fnm1 = state_prev.f
        y_predict = state_predict.y
        fn = self.f(self.t, y_predict)
        y_correct = adams_moulton(ynm1, fn, fnm1, hn)
        return StepState(
            t=self.t,
            h=hn,
            y=y_correct,
            f=self.f(self.t, y_correct)
        )

    def _predict(self,hn,state_prev,state_prev2=None,first_step=False):
        if first_step:
            y_predict = adams_bashforth_1(state_prev.y, state_prev.f, hn)
        else:
            y_predict = adams_bashforth_2(state_prev.y, state_prev.f, state_prev2.f, hn, state_prev.h)
        return StepState(
            t=self.t,
            h=hn,
            y=y_predict,
            f=self.f(self.t,y_predict)
        )

    def take_first_step(self, hn):
        state_prev = self.step_states[-1]
        state_predict = self._predict(hn,state_prev,first_step=True)
        state_correct = self._correct(state_prev, state_predict, hn)
        return state_predict,state_correct

    def take_step(self, hn):
        state_prev  = self.step_states[-1]
        state_prev2 = self.step_states[-2]
        state_predict = self._predict(hn,state_prev,state_prev2)
        state_correct = self._correct(state_prev, state_predict, hn)
        return state_predict,state_correct

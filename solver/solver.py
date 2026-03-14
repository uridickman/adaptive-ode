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
            rtol        : float=1e-3,
            h           : float=-1
        ):
        
        self.tmin,self.tmax = trange
        self.rtol = rtol
        self.f = f
        self.h = h
        self.step_states = []
        

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
        self.t += self.h
        self.advance_first_step()
        while self.t < self.tmax:
            self.t += self.h
            self.advance()
        self.unpack_step_states()


    def advance_first_step(self):
        state_prev = self.step_states[-1]
        ynm1 = state_prev.y
        fnm1 = state_prev.f
        hn = self.h
        y_predict = adams_bashforth_1(ynm1,fnm1,hn)

        state_next = StepState(
            self.t,
            self.h,
            self.f(self.t,y_predict),
            y_predict
        )
        fn = state_next.f
        y_correct = adams_moulton(ynm1,fn,fnm1,hn)
        state_next.y = y_correct
        state_next.f = self.f(self.t,y_correct)

        self.step_states.append(state_next)


    def advance(self):
        state_prev = self.step_states[-1]
        state_prev2 = self.step_states[-2]
        ynm1 = state_prev.y
        fnm1 = state_prev.f
        fnm2 = state_prev2.f
        hn = self.h
        hnm1 = state_prev.h
        y_predict = adams_bashforth_2(ynm1,fnm1,fnm2,hn,hnm1)

        state_next = StepState(
            self.t,
            self.h,
            self.f(self.t,y_predict),
            y_predict
        )
        fn = state_next.f
        y_correct = adams_moulton(ynm1,fn,fnm1,hn)
        state_next.y = y_correct
        state_next.f = self.f(self.t,y_correct)

        self.step_states.append(state_next)

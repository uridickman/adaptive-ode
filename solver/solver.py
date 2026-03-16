import numpy as np
from .jit_functions import *
from typing import Tuple,Callable
from collections import deque

class ODESolver:
    def __init__(
            self,
            f           : Callable,
            y0          : np.ndarray,
            trange      : Tuple[float,float],
            t_eval      : np.ndarray=None,
            etol        : float=1e-3,
            h           : float=-1
            
        ):
        
        self.tmin,self.tmax = trange
        self.f = f
        self.h = 0.1
        self.is_adaptive = h < 0

        self.step_states = []

        self._etol = etol
        self._frac = 0.9
        self._p = 2
        self._max_iter = 15

        step0 = StepState(
            self.tmin,
            self.h,
            self.f(self.tmin,y0).astype(np.float64),
            y0.astype(np.float64)
        )

        self.step_states.append(step0)
        self.step_states_at_teval = []
        self.t = float(self.tmin)

        if self.is_adaptive:
            self.h = self.predict_step_size(first_step=True)
            step0.h = self.h
        else:
            self.h = h

        if t_eval is not None and len(t_eval) > 0:
            self.interpolate_t = True
            self.t_eval_queue = deque(t_eval.astype(np.float64))
        else:
            self.interpolate_t = False
        

    def solve(self):
        if self.interpolate_t:
            self.solve_at_teval(adaptive=self.is_adaptive)
        else:
            self.solve_no_teval(adaptive=self.is_adaptive)
            

    def solve_at_teval(self,adaptive=False):
        _, state_next = self.take_first_step(self.h)
        self.step_states.append(state_next)
        t_current_eval = self.t_eval_queue.popleft()
        while self.t_eval_queue:
            t_current_eval = self.t_eval_queue[0]

            if adaptive:
                self.h = self.predict_step_size()
            self.t += self.h
            _, state_next = self.take_step(self.h)
            self.step_states.append(state_next)

            while self.step_states[-1].t >= t_current_eval:
                self.t_eval_queue.popleft()
                state_interp = self.interpolate_state(t_current_eval)
                self.step_states_at_teval.append(state_interp)
                if not self.t_eval_queue:
                    break
                t_current_eval = self.t_eval_queue[0]
        self.T, self.Y, self.F, self.H = unpack_step_states(self.step_states_at_teval)


    def interpolate_state(self,t_current_eval):
        if t_current_eval == self.t:
            return self.step_states[-1]
        num_pts = min(len(self.step_states), 3)
        t_interp = np.array([s.t for s in self.step_states[-num_pts:]], dtype=np.float64)
        y_interp = np.array([s.y for s in self.step_states[-num_pts:]], dtype=np.float64)
        _,n = y_interp.shape
        y_current_eval = np.empty(n, dtype=np.float64)
        for j in range(n):
            y_data = y_interp[:,j]
            y_eval = interpolate(t_interp,y_data,np.array([t_current_eval],dtype=np.float64))
            y_current_eval[j] = y_eval

        state_interp = StepState(
            t_current_eval,
            -1,
            self.f(t_current_eval,y_current_eval),
            y_current_eval
        )

        return state_interp


    def solve_no_teval(self, adaptive=False):
        _, state_next = self.take_first_step(self.h)
        self.step_states.append(state_next)

        while self.t < self.tmax:
            if adaptive:
                self.h = self.predict_step_size()
            self.t += self.h
            _, state_next = self.take_step(self.h)
            self.step_states.append(state_next)
    
        self.T, self.Y, self.F, self.H = unpack_step_states(self.step_states)


    def predict_step_size(self,first_step=False):
        h = self.h
        
        for _ in range(self._max_iter):
            state_predict,state_correct = self.take_first_step(h) if first_step else self.take_step(h)
            est = lte(state_predict.y,state_correct.y)
            if est <= self._etol:
                return h
            h = new_step_size(est,h,self._frac,self._etol,self._p)

        raise RuntimeError("Step size did not converge.")

    def take_first_step(self, hn):
        state_prev = self.step_states[-1]
        state_predict = predict_first(self.t,hn,self.f,state_prev)
        state_correct = correct(self.t,hn,self.f,state_prev, state_predict)
        return state_predict,state_correct

    def take_step(self, hn):
        state_prev  = self.step_states[-1]
        state_prev2 = self.step_states[-2]
        state_predict = predict(self.t,hn,self.f,state_prev,state_prev2)
        state_correct = correct(self.t,hn,self.f,state_prev, state_predict)
        return state_predict,state_correct
    

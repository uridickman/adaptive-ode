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
        self._max_iter = 6

        step0 = StepState(
            self.tmin,
            self.h,
            self.f(self.tmin,y0).astype(np.float64),
            y0.astype(np.float64),
            0
        )

        self.step_states.append(step0)          # All states at times determined by self.h
        self.step_states_at_teval = []          # States at times determined by t_eval
        self.t = float(self.tmin)               # Timestep of the current step


        # Predict the first step size
        if self.is_adaptive:
            self.h,num_iter_first = self.predict_step_size(True)
            step0.h = self.h
            step0.num_iter = num_iter_first
        else:
            self.h = h

        # If evaluating at specific times, then create a queue
        # with the times at which to evaluate the solution.
        # They will be interpolated using an order 3 interpolating polynomial
        if t_eval is not None and len(t_eval) > 0:
            self.interpolate_t = True
            self.t_eval_queue = deque(t_eval.astype(np.float64))
        else:
            self.interpolate_t = False
        

    def solve(self):
        """Execute the solving sequences depending on if evaluating
        the solution at specific times or not.
        """
        if self.interpolate_t:
            self.solve_at_teval(adaptive=self.is_adaptive)
        else:
            self.solve_no_teval(adaptive=self.is_adaptive)
            

    def solve_at_teval(self,adaptive=False):
        _, state_next = take_first_step(self.step_states[-1],self.t,self.h,self.f,0)
        self.step_states.append(state_next)
        t_current_eval = self.t_eval_queue.popleft()
        while self.t_eval_queue:
            t_current_eval = self.t_eval_queue[0]

            if adaptive:
                self.h,num_iter = self.predict_step_size()
            self.t += self.h
            _, state_next = take_step(self.step_states[-1],self.step_states[-2],self.t,self.h,self.f,num_iter)
            self.step_states.append(state_next)

            while self.t >= t_current_eval:
                self.t_eval_queue.popleft()
                state_interp = self.interpolate_state(t_current_eval)
                state_interp.num_iter = 0
                self.step_states_at_teval.append(state_interp)
                if not self.t_eval_queue:
                    break
                t_current_eval = self.t_eval_queue[0]
        self.T, self.Y, self.F, self.H, self.N = unpack_step_states(self.step_states_at_teval)


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
            self.h,
            self.f(t_current_eval,y_current_eval),
            y_current_eval,
            0
        )

        return state_interp

    def solve_no_teval(self, adaptive=False):
        _, state_next = take_first_step(self.step_states[-1],self.t,self.h,self.f,0)
        self.step_states.append(state_next)

        while self.t < self.tmax:
            if adaptive:
                self.h,num_iter = self.predict_step_size()
            self.t += self.h
            _, state_next = take_step(self.step_states[-1],self.step_states[-2],self.t,self.h,self.f,num_iter)
            self.step_states.append(state_next)
    
        self.T, self.Y, self.F, self.H, self.N = unpack_step_states(self.step_states)

    def predict_step_size(self,first_step=False):
        args = (self.h,
                self.t,
                self.f,
                self._etol,
                self._frac,
                self._p,
                self._max_iter
            )
        if first_step:
            return predict_first_step_size(
                        self.step_states[-1],
                        *args
                    )
        
        return predict_step_size(
                    self.step_states[-1],
                    self.step_states[-2],
                    *args
                )
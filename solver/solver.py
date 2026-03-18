import numpy as np
import numpy.typing as npt
from .jit_functions import *
from typing import Tuple,Callable
from collections import deque


class ODESolver:
    def __init__(
            self,
            f           : Callable[
                                [float | npt.NDArray[np.float64], npt.NDArray[np.float64]],
                                npt.NDArray[np.float64]
                        ],
            y0          : npt.NDArray[np.float64],
            trange      : Tuple[np.float64,np.float64],
            t_eval      : npt.NDArray[np.float64]=np.array([]),
            etol        : np.float64=1e-3,
            h           : np.float64=-1
            
        ):
        
        self.tmin,self.tmax = trange
        self.f = f
        
        self.is_adaptive = h < 0
        self.h = 0.1 if self.is_adaptive else h
        self.t_eval_input = np.sort(t_eval.astype(np.float64))
        self.y0 = y0.astype(np.float64)

        self._etol = etol
        self._frac = 0.9
        self._p = 2
        self._max_iter = 100
        

    def initialize(self):
        """Initialize the solver: set up the step states list, initialize the
        first step size.
        """
        self.step_states = []
        step0 = StepState(
            self.tmin,
            self.h,
            self.f(self.tmin,self.y0).astype(np.float64),
            self.y0,
            0
        )

        self.step_states.append(step0)          # All states at times determined by self.h
        self.step_states_at_teval = []          # States at times determined by t_eval
        self.t = float(self.tmin)               # Timestep of the current step


        # Predict the first step size
        if self.is_adaptive:
            self.h,self.h_next,num_iter_first = self.predict_step_size(self.h,True)
            step0.h = self.h
            step0.num_iter = num_iter_first

        # If evaluating at specific times, then create a queue
        # with the times at which to evaluate the solution.
        # They will be interpolated using an order 3 interpolating polynomial
        if len(self.t_eval_input) > 0:
            self.interpolate_t = True
            self.t_eval_queue = deque(self.t_eval_input)
        else:
            self.interpolate_t = False

        self.solve_first_step()


    def solve(self):
        """Execute the solving sequences depending on if evaluating
        the solution at specific times or not.
        """
        self.initialize()
        if self.interpolate_t:
            self.solve_at_teval(adaptive=self.is_adaptive)
        else:
            self.solve_no_teval(adaptive=self.is_adaptive)
            

    def solve_first_step(self):
        """Solve the first step using 1st order PECE
        """
        _, state_next = take_first_step(self.step_states[-1],self.t,self.h,self.f,0)
        self.step_states.append(state_next)


    def solve_at_teval(self,adaptive:bool=False):
        """Solve for all t<tmax or until self.t_eval_queue is empty,
        interpolating the solutions at the times in self.t_eval_queue.
        Uses 2nd order PECE.

        Args:
            adaptive (bool, optional): If adaptive, predicts step size
            for error control. Defaults to False.
        """
        t_current_eval = self.t_eval_queue.popleft()
        while self.t_eval_queue or self.t <= self.tmax:
            if adaptive:
                self.h,self.h_next,num_iter = self.predict_step_size(self.h_next)
            else:
                num_iter = 0
            self.t += self.h
            _, state_next = take_step(self.step_states[-1],self.step_states[-2],self.t,self.h,self.f,num_iter)
            self.step_states.append(state_next)
            
            while self.t >= t_current_eval:
                state_interp = self.interpolate_state(t_current_eval)
                state_interp.num_iter = 0
                self.step_states_at_teval.append(state_interp)
                if not self.t_eval_queue:
                    break
                t_current_eval = self.t_eval_queue.popleft()
        self.T, self.Y, self.F, self.H, self.N = unpack_step_states(self.step_states_at_teval)


    def interpolate_state(self,t_current_eval:float):
        """Interpolates the state at a particular time.

        Args:
            t_current_eval (float): time at which to interpolate the state.

        Returns:
            StepState: Interpolated state.
        """
        if t_current_eval == self.tmin:
            return self.step_states[0]
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


    def solve_no_teval(self, adaptive:bool=False):
        """Solve for all t using 2nd order PECE.

        Args:
            adaptive (bool, optional): If adaptive, then predict the
            step size for error control. Defaults to False.
        """
        while self.t < self.tmax:
            if adaptive:
                self.h,self.h_next,num_iter = self.predict_step_size(self.h_next)
            else:
                num_iter = 0
            self.t += self.h
            _, state_next = take_step(self.step_states[-1],self.step_states[-2],self.t,self.h,self.f,num_iter)
            self.step_states.append(state_next)
    
        self.T, self.Y, self.F, self.H, self.N = unpack_step_states(self.step_states)


    def predict_step_size(self,h0:np.float64,first_step:bool=False):
        """Predict the step size so that the estimated error is below self._etol.

        Args:
            h0 (float): starting timestep
            first_step (bool, optional): if first_step, then predict step size 
            using 1st order PECE. Otherwise, use 2nd order PECE. Defaults to False.

        Returns:
            float: the new step size.
        """
        args = (h0,
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
import numpy as np
from .jit_functions import *
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

    def adams_bashforth_step(self,i):
        ynm1 = self.solution_list[i-1]
        ynm2 = self.solution_list[i-2]
        tnm1 = self.T[i-1]
        tnm2 = self.T[i-2]
        hn = self.H[i]
        hnm1 = self.H[i-1]
        fnm1 = self.f(tnm1,ynm1)
        fnm2 = self.f(tnm2,ynm2)

        return ynm1 + hn * fnm1 \
                + (fnm1 - fnm2) / hnm1 * hn*hn / 2


    def interpolate_sym(
            self,
            xsym        :sp.Symbol,
            x_data      :np.ndarray,
            y_data      :np.ndarray
        )               -> np.ndarray:
        
        n = len(x_data)
        coeffs = newton_coeffs(x_data,y_data,order=n)

        out    = coeffs[0]
        mult_x = 1

        for k in range(1, n):
            mult_x = mult_x * (xsym - x_data[k - 1])
            out = out + coeffs[k] * mult_x

        return out

    def integral_interpolant(self,xquery,x,y):
        xsym = sp.Symbol('x')
        poly_np = self.interpolate_sym(xsym,x,y)
        poly_int = sp.integrate(poly_np)
        poly_int_np = sp.lambdify(xsym, poly_int, modules="numpy")
        return poly_int_np(xquery)

    def compute_step_size(self):
        num_iter = 0
        while num_iter < self._max_iter:
            pass
        if num_iter >= self._max_iter:
            warn("Step size failed to converge.",category=RuntimeWarning)
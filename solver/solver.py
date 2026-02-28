import numpy as np
from .jit_functions import *
from jax import jacfwd
import jax.numpy as jnp
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
        self.t = self.tmin
        self.rtol = rtol
        self.y_at = np.sort(y_at)

        self.max_order = 5
        self.f = f
        self.H = [self.h]
        self.T = [self.t]

        self.jacobian = jacfwd(self.g_adams_moulton)

        self.solution_list = [self.Y0]

        self._max_iter = 5

    def solve(self):
        order = 1
        while t < self.tmax:
            t = self.T[-1]
            if self.use_adaptive_h:
                self.h = self.compute_step_size()
                self.H.append(self.h)

            self.take_step(self.solution_list[-1],self.h,order)

            t += self.h
            self.T.append(t)
            
        self.Y = np.array(self.solution_list)
        

    def take_step(self,n,order):
        y_predict = self.adams_bashforth_step(order)
        try:
            y_correct = self.adams_moulton_step(y_predict,order)
        except RuntimeError:
            warn(f"Newton iteration failed to converge on step {n}")
        return y_correct


    def adams_moulton_step(
            self,
            y_pred, # Predicted by Adams-Bashforth
            order
        ):
        t_eval = self.t + self.h
        y_next = self.newton_step(
                        self.f,
                        self.g_adams_moulton,
                        t_eval,
                        y_pred,
                        self.rtol
                    )
        t_nodes = np.asarray(self.T[-order:] + [t_eval])
        y_nodes = np.asarray(self.solution_list[-order:] + [y_next])
        f_nodes = self.f(t_nodes, y_nodes)

        delta_y = np.zeros_like(y_next)

        for f_val in f_nodes:
            delta_y += self.integral_interpolant(self.t,t_eval,t_nodes,f_val)
            
        return y_next[-1] + delta_y


    def g_adams_moulton(self,Y,order):
        t_eval = self.t + self.h
        y_prev = self.solution_list[-1]

        t_nodes = np.asarray(self.T[-order:] + [t_eval])
        y_nodes = np.asarray(self.solution_list[-order:] + [Y])
        f_nodes = self.f(t_nodes, y_nodes)

        delta_y = np.zeros_like(Y)

        for f_val in f_nodes:
            delta_y += self.integral_interpolant(self.t,t_eval,t_nodes,f_val)
            
        return Y - y_prev - delta_y


    def adams_bashforth_step(self, order):
        t_nodes = np.asarray(self.T[-order-1:-1])
        y_nodes = np.asarray(self.solution_list[-order-1:-1])

        f_nodes = self.f(t_nodes, y_nodes)

        delta_y = np.zeros_like(y_nodes[-1])
        t_eval = self.t + self.h

        for f_val in f_nodes:
            delta_y += self.integral_interpolant(self.t,t_eval,t_nodes,f_val)
            
        return y_nodes[-1] + delta_y


    def adams_bashforth2_(self,i):
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


    def evaluate_Jacobian(self,yeval:np.ndarray):
        self.jacobian = jacfwd(self.g_adams_moulton)
        jyeval = jnp.asarray(yeval)
        return np.array(self.jacobian(jyeval))


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


    def integral_interpolant(self,x1,x2,x,y):
        xsym = sp.Symbol('x')
        poly_np = self.interpolate_sym(xsym,x,y)
        poly_int = sp.integrate(poly_np)
        poly_int_np = sp.lambdify(xsym, poly_int, modules="numpy")
        return poly_int_np(x2) - poly_int_np(x1)


    def newton_step(self,f,g,t,yprev):
        diff = np.inf
        max_iter = 3

        num_iter = 0

        Y = yprev

        while diff > self.rtol and num_iter <= max_iter:
            dGdy = self.evaluate_Jacobian(Y)

            rhs = -g(f,t,yprev,Y)
            delta = np.linalg.solve(dGdy,rhs)
            ynext = Y + delta

            diff = np.linalg.norm(delta,order=1)
            Y = ynext
            num_iter = num_iter + 1

            if num_iter > max_iter:
                raise RuntimeError


    def compute_etol(self):
        pass


    def compute_step_size(self):
        num_iter = 0
        while num_iter < self._max_iter:
            pass


        if num_iter >= self._max_iter:
            self.jacobian = self.compute_Jacobian(self.solution_list[-1])
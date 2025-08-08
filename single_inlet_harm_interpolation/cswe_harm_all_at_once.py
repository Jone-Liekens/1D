

import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt, zeros, ones, nan
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm

from plots import *

class CSWEHarm():

    def __init__(self):

        self.debug = False

        self.A = 0.72
        self.H = 7.12 
        self.L = 8e3
        
        self.g = 9.81
        self.sigma = 1.4e-4
        self.rho_w = 1025
        self.rho_s = 2650

        self.h0 = 0.0025
        self.p = 0.4 # porosity
        self.C_D = 0.0025
        self.lmbda = 6.8e-6
        self.d50 = 0.13e-3
        
        self.a_r = 1e-2
        self.r = 0.24

        self.iota = 1

        self.small_number = nan

        self.set_derivative_vars()
    
    def set_derivative_vars(self):
        self.epsilon = self.A / self.H
        self.eta = self.sigma * self.L / sqrt(self.g * self.H)
        self.U = self.epsilon * self.sigma * self.L
        self.kappa = self.g * self.H / (self.sigma * self.L) ** 2

        self.s = self.rho_s / self.rho_w

        self.delta = 0.04 * self.C_D**(3/2) * self.A * (self.sigma * self.L)**4 / \
                    (self.g**2 * (self.s-1)**2 * self.d50 * self.H**6 * (1-self.p))

    def h(self, x):
        return x
        # return 0.9*x

    def h_x(self, x):
        return 1
        # return 0.9

    def solve(self):
        
        def deriv(x, y):
            dz0_c, dz0_s, u0_c, u0_s, dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = y
            h, h_x = self.h(x), self.h_x(x)

            # Leading order
            d_dz0_c = 1 / self.kappa * ( - self.r / (1 - h + self.h0) * u0_c - u0_s)
            d_dz0_s = 1 / self.kappa * ( - self.r / (1 - h + self.h0) * u0_s + u0_c)

            d_u0_c = (-dz0_s + u0_c * h_x)  / (1 - h + self.small_number)
            d_u0_s = ( dz0_c + u0_s * h_x)  / (1 - h + self.small_number) 

            # First order
            d_dz1_r = (1 / (1 - h + self.h0) * (
                - self.r * u1_r
                - 1 / 2 * (  dz0_c *  u0_s - dz0_s *  u0_c)
                - 1 / 2 * (  dz0_s * d_dz0_s + dz0_c * d_dz0_c) * self.kappa
            ) - 1/2 * (u0_c * d_u0_c + u0_s * d_u0_s)) / self.kappa
            d_dz1_c = (1 / (1 - h + self.h0) * (
                - self.r * u1_c
                - 1 / 2 * (  dz0_c *  u0_s + dz0_s *  u0_c)
                - 1 / 2 * (- dz0_s * d_dz0_s + dz0_c * d_dz0_c) * self.kappa
            ) - 1/2 * (u0_c * d_u0_c - u0_s * d_u0_s) - 2 * u1_s) / self.kappa
            d_dz1_s = (1 / (1 - h + self.h0) * (
                - self.r * u1_s
                + 1 / 2 * (  dz0_c *  u0_c - dz0_s *  u0_s)
                - 1 / 2 * (  dz0_c * d_dz0_s + dz0_s * d_dz0_c) * self.kappa
            ) - 1/2 * (u0_c * d_u0_s + u0_s * d_u0_c) + 2 * u1_c) / self.kappa

            d_u1_r = 1 / (1 - h + self.small_number) * (h_x * u1_r             - 1 / 2 * (dz0_c * d_u0_c + d_dz0_c * u0_c + dz0_s * d_u0_s + d_dz0_s * u0_s))
            d_u1_c = 1 / (1 - h + self.small_number) * (h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * d_u0_c + d_dz0_c * u0_c - dz0_s * d_u0_s - d_dz0_s * u0_s))
            d_u1_s = 1 / (1 - h + self.small_number) * (h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * d_u0_c + d_dz0_s * u0_c + dz0_c * d_u0_s + d_dz0_c * u0_s))

            return [d_dz0_c, d_dz0_s, d_u0_c, d_u0_s, d_dz1_r, d_dz1_c, d_dz1_s, d_u1_r, d_u1_c, d_u1_s]

 

        def bc(y_left, y_right):
            dz0_c_l, dz0_s_l, u0_c_l, u0_s_l, dz1_r_l, dz1_c_l, dz1_s_l, u1_r_l, u1_c_l, u1_s_l = y_left
            dz0_c_r, dz0_s_r, u0_c_r, u0_s_r, dz1_r_r, dz1_c_r, dz1_s_r, u1_r_r, u1_c_r, u1_s_r = y_right

            h_r, h_x_r = self.h(1), self.h_x(1)

            d_dz0_c_r = 1 / self.kappa * ( - self.r / (1 - h_r + self.h0) * u0_c_r - u0_s_r)
            d_dz0_s_r = 1 / self.kappa * ( - self.r / (1 - h_r + self.h0) * u0_s_r + u0_c_r)

            d_u0_c_r = (-dz0_s_r + u0_c_r * h_x_r)  / (1 - h_r + self.small_number)
            d_u0_s_r = ( dz0_c_r + u0_s_r * h_x_r)  / (1 - h_r + self.small_number) 

            return [
                dz0_c_l - 1,
                dz0_s_l,
                dz0_s_r - h_x_r * u0_c_r,
                dz0_c_r + h_x_r * u0_s_r,
                dz1_r_l, 
                dz1_s_l, 
                dz1_c_l,
                h_x_r * u1_r_r - 1 / 2 * (dz0_c_r * d_u0_c_r + d_dz0_c_r * u0_c_r + dz0_s_r * d_u0_s_r + d_dz0_s_r * u0_s_r),
                h_x_r * u1_c_r - 1 / 2 * (dz0_c_r * d_u0_c_r + d_dz0_c_r * u0_c_r - dz0_s_r * d_u0_s_r - d_dz0_s_r * u0_s_r) - 2 * dz1_s_r,
                h_x_r * u1_s_r - 1 / 2 * (dz0_c_r * d_u0_s_r + d_dz0_c_r * u0_s_r + dz0_s_r * d_u0_c_r + d_dz0_s_r * u0_c_r) + 2 * dz1_c_r
            ]
        
        self.x = linspace(0, 1, 1000)
        vector_guess = 0.1 * np.ones((10, len(self.x)))

        sol = scipy.integrate.solve_bvp(deriv, bc, self.x, vector_guess, tol=1e-4, max_nodes=20000)
        
        if sol.status or self.debug:
            print(sol)
            raise SystemError
        
        self.y = sol

    def solve_with_h(self):
        
        def deriv(x, y):
            dz0_c, dz0_s, u0_c, u0_s, dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s, h, h_x = y

            # Leading order
            d_dz0_c = 1 / self.kappa * ( - self.r / (1 - h + self.h0) * u0_c - u0_s)
            d_dz0_s = 1 / self.kappa * ( - self.r / (1 - h + self.h0) * u0_s + u0_c)

            d_u0_c = (-dz0_s + u0_c * h_x)  / (1 - h + self.small_number)
            d_u0_s = ( dz0_c + u0_s * h_x)  / (1 - h + self.small_number) 

            # First order
            d_dz1_r = (1 / (1 - h + self.h0) * (
                - self.r * u1_r
                - 1 / 2 * (  dz0_c *  u0_s - dz0_s *  u0_c)
                - 1 / 2 * (  dz0_s * d_dz0_s + dz0_c * d_dz0_c) * self.kappa
            ) - 1/2 * (u0_c * d_u0_c + u0_s * d_u0_s)) / self.kappa
            d_dz1_c = (1 / (1 - h + self.h0) * (
                - self.r * u1_c
                - 1 / 2 * (  dz0_c *  u0_s + dz0_s *  u0_c)
                - 1 / 2 * (- dz0_s * d_dz0_s + dz0_c * d_dz0_c) * self.kappa
            ) - 1/2 * (u0_c * d_u0_c - u0_s * d_u0_s) - 2 * u1_s) / self.kappa
            d_dz1_s = (1 / (1 - h + self.h0) * (
                - self.r * u1_s
                + 1 / 2 * (  dz0_c *  u0_c - dz0_s *  u0_s)
                - 1 / 2 * (  dz0_c * d_dz0_s + dz0_s * d_dz0_c) * self.kappa
            ) - 1/2 * (u0_c * d_u0_s + u0_s * d_u0_c) + 2 * u1_c) / self.kappa

            d_u1_r = 1 / (1 - h + self.small_number) * (h_x * u1_r             - 1 / 2 * (dz0_c * d_u0_c + d_dz0_c * u0_c + dz0_s * d_u0_s + d_dz0_s * u0_s))
            d_u1_c = 1 / (1 - h + self.small_number) * (h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * d_u0_c + d_dz0_c * u0_c - dz0_s * d_u0_s - d_dz0_s * u0_s))
            d_u1_s = 1 / (1 - h + self.small_number) * (h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * d_u0_c + d_dz0_s * u0_c + dz0_c * d_u0_s + d_dz0_c * u0_s))

            # Morphodynamics
            d_h = h_x 

            u  = self.epsilon * (  u1_r + (u0_c**2 + u0_s**2))
            du = self.epsilon * (d_u1_r + (2 * u0_c * d_u0_c + 2 * u0_s * d_u0_s))
            d_h_x = self.iota * self.delta / self.lmbda * u ** 4 * du

            return [d_dz0_c, d_dz0_s, d_u0_c, d_u0_s, d_dz1_r, d_dz1_c, d_dz1_s, d_u1_r, d_u1_c, d_u1_s, d_h, d_h_x]

 

        def bc(y_left, y_right):
            dz0_c_l, dz0_s_l, u0_c_l, u0_s_l, dz1_r_l, dz1_c_l, dz1_s_l, u1_r_l, u1_c_l, u1_s_l, h_l, h_x_l = y_left
            dz0_c_r, dz0_s_r, u0_c_r, u0_s_r, dz1_r_r, dz1_c_r, dz1_s_r, u1_r_r, u1_c_r, u1_s_r, h_r, h_x_r = y_right

            # recompute leading order derivatives
            d_dz0_c_r = 1 / self.kappa * ( - self.r / (1 - h_r + self.h0) * u0_c_r - u0_s_r)
            d_dz0_s_r = 1 / self.kappa * ( - self.r / (1 - h_r + self.h0) * u0_s_r + u0_c_r)
            d_u0_c_r = (-dz0_s_r + u0_c_r * h_x_r)  / (1 - h_r + self.small_number)
            d_u0_s_r = ( dz0_c_r + u0_s_r * h_x_r)  / (1 - h_r + self.small_number) 

            return [
                # Leading order
                dz0_c_l - 1,
                dz0_s_l,
                dz0_s_r - h_x_r * u0_c_r,
                dz0_c_r + h_x_r * u0_s_r,

                # First order
                dz1_r_l, 
                dz1_s_l, 
                dz1_c_l,
                h_x_r * u1_r_r - 1 / 2 * (dz0_c_r * d_u0_c_r + d_dz0_c_r * u0_c_r + dz0_s_r * d_u0_s_r + d_dz0_s_r * u0_s_r),
                h_x_r * u1_c_r - 1 / 2 * (dz0_c_r * d_u0_c_r + d_dz0_c_r * u0_c_r - dz0_s_r * d_u0_s_r - d_dz0_s_r * u0_s_r) - 2 * dz1_s_r,
                h_x_r * u1_s_r - 1 / 2 * (dz0_c_r * d_u0_s_r + d_dz0_c_r * u0_s_r + dz0_s_r * d_u0_c_r + d_dz0_s_r * u0_c_r) + 2 * dz1_c_r,

                # Morphodynamics
                h_l,
                h_r - 1
            ]
        
        self.x = linspace(0, 1, 1000)
        vector_guess = 0.1 * np.ones((12, len(self.x)))

        sol = scipy.integrate.solve_bvp(deriv, bc, self.x, vector_guess, tol=1e-3, max_nodes=20000)
        
        if sol.status or self.debug:
            print(sol)
            raise SystemError
        
        self.y = sol


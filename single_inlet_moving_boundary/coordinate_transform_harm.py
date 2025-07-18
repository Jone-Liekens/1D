



import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm

from single_inlet_moving_boundary.plots import *
from single_inlet_moving_boundary.coordinate_transform_solution import CoordinateTransformSolution

class CoordinateTransformHarm():

    def __init__(self):
        
        self.A = 0.84
        
        self.H = 12
        self.L = 1.9e4
        self.h0 = 0.0025
        # h0 = 0.00025
        self.omega = 1.4e-4

        self.r = 0.24

        self.epsilon = self.A / self.H


        self.kappa = 1.7e1 * self.epsilon


    def generate_solution(self):

        print(self.epsilon)
        self.l_t = 1 + self.epsilon * self.lmbda0_t
        # self.D_xt = 1 + self.epsilon * self.dz0_xt - self.h_xt - 1
        self.D_xt = self.epsilon * self.dz0_xt + (1 - self.x[:, np.newaxis])



        plt.plot(self.epsilon * self.dz0_xt[400, :])
        plt.plot(self.h_xt[0, :])
        plt.show()
        self.u_xt = self.epsilon * self.u0_xt 
        print(self.u_xt.shape, self.D_xt.shape, self.l_t.shape)
        print(self.x.shape, self.t.shape)
        
        return CoordinateTransformSolution(self.t, self.x, self.l_t, self.D_xt, self.u_xt)

    
    def solve(self):
        self.solve_LO

    def h(self, x):
        return x

    def h_x(self, x):
        return 1
    
    def h_xx(self, x):
        return 0

    def solve_LO(self):

        def deriv(x, vector, lmbda):
            u_s, u_c, dz_s, dz_c = vector
            lmbda_s, lmbda_c = lmbda

            h = self.h(x)
            h_x = self.h_x(x)

            du_s = 1 / (1 - h + self.h0) * ( dz_c + x * h_x * lmbda_c + u_s * h_x)
            du_c = 1 / (1 - h + self.h0) * (-dz_s + x * h_x * lmbda_s + u_c * h_x)

            friction_term = - self.r / (1 - h + self.h0)

            ddz_s = 1 / self.kappa * (friction_term * u_s + u_c)
            ddz_c = 1 / self.kappa * (friction_term * u_c - u_s)

            return [du_s, du_c, ddz_s, ddz_c]
        

        def bc(vector_left, vector_right, lmbda):
            u_s0, u_c0, dz_s0, dz_c0 = vector_left
            u_s1, u_c1, dz_s1, dz_c1 = vector_right
            lmbda_s, lmbda_c = lmbda

            return [dz_s0, dz_c0 - 1, dz_s1, dz_c1, lmbda_c + u_s1, lmbda_s - u_c1]
        
        self.x = linspace(0, 1, 1000)
        vector_guess = 1e-3 * np.ones((4, 1000))

        # p = lambda initial guess
        sol = scipy.integrate.solve_bvp(deriv, bc, self.x, vector_guess, p=[0.1, 0.1], tol=1e-4, max_nodes=5000)
        print(sol)
        
        self.x = sol.x
        
        self.u0_s, self.u0_c, self.dz0_s, self.dz0_c = sol.y
        self.lmbda0_s, self.lmbda0_c = sol.p

        self.u0_s_dx, self.u0_c_dx, self.dz0_s_dx, self.dz0_c_dx = deriv(self.x, sol.y, sol.p)


        # for plotting: store numerical values
        self.t = np.linspace(0, 6*pi, 1000)
        self.u0_xt = self.u0_s[:, np.newaxis] * sin(self.t[np.newaxis, :]) + \
            self.u0_c[:, np.newaxis] * cos(self.t[np.newaxis, :])
        self.dz0_xt = self.dz0_s[:, np.newaxis] * sin(self.t[np.newaxis, :]) + \
            self.dz0_c[:, np.newaxis] * cos(self.t[np.newaxis, :])
        self.lmbda0_t = self.lmbda0_s * sin(self.t) + self.lmbda0_c * cos(self.t)
        self.h_xt = self.x[:, np.newaxis]* self.lmbda0_t[np.newaxis, :]




    def solve_FO(self):

        def deriv(x, vector, lmbda):


            ld0_s, ld0_c, dz0_s, dz0_c, u0_s, u0_c = self.ld0_s, self.ld0_c, self.dz0_s, self.dz0_c, self.u0_s, self.u0_c 

            dz0_s_dx, dz0_c_dx, u0_s_dx, u0_c_dx = \
                self.dz0_s_dx, self.dz0_c_dx, self.u0_s_dx, self.u0_c_dx 


            u1_r, dz1_r, u1_s, u1_c, dz1_s, dz1_c = vector
            ld1_r, ld1_s, ld1_c = lmbda

            h, h_x, h_xx = self.h(x), self.h_x(x), self.h_xx(x)


            du1_r = 1 / (1 - h + self.h0) * (
                h_x * u1_r - 1 / 2 * (
                    + (ld0_c * dz0_s - ld0_s * dz0_c) * (-1)
                    + (dz0_c * ld0_s - dz0_s * ld0_c) * (x)
                    + ( u0_c * )
                )
            ) 

            

            du_s = 1 / (1 - h + self.h0) * ( dz_c + x * h_x * lmbda_c + u_s * h_x)
            du_c = 1 / (1 - h + self.h0) * (-dz_s + x * h_x * lmbda_s + u_c * h_x)

            friction_term = - self.r / (1 - h + self.h0)

            ddz_s = 1 / self.kappa * (friction_term * u_s + u_c)
            ddz_c = 1 / self.kappa * (friction_term * u_c - u_s)

            return [du_s, du_c, ddz_s, ddz_c]
        

        def bc(vector_left, vector_right, lmbda):
            u_s0, u_c0, dz_s0, dz_c0 = vector_left
            u_s1, u_c1, dz_s1, dz_c1 = vector_right
            lmbda_s, lmbda_c = lmbda

            return [dz_s0, dz_c0 - 1, dz_s1, dz_c1, lmbda_c + u_s1, lmbda_s - u_c1]
        
        self.x = linspace(0, 1, 1000)
        vector_guess = 0.1 * np.ones((4, 1000))

        sol = scipy.integrate.solve_bvp(deriv, bc, self.x, vector_guess, p = [0.1, 0.1])
        print(sol)

        self.u0_s, self.u0_c, self.dz0_s, self.dz0_c = sol.y
        self.lmbda0_s, self.lmbda0_c = sol.p


if __name__ == "__main__":
    computation = CoordinateTransformHarm()
    computation.solve_LO()
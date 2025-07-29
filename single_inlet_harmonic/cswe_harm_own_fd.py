


# use (?)
# from numba import jit  
# @jit 

import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt, zeros
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate

from plots import *
from cswe_harm_solution import CSWEHarmSolution


class LO_own():
    def __init__(self, deriv):
        self.deriv = deriv
        self.nx = 500
    
    def solve(self):
        x = np.linspace(0, 1, self.nx)
        self.dx = x[1] - x[0]
        initial_guess = 0.1 * np.ones((4, len(x)))
        
        dy = 1e50 * np.ones(initial_guess.shape)
        y = initial_guess

        while np.linalg.norm(dy) > 1e-12:
            print(np.linalg.norm(dy))
            dy = self.solve_iteration(x, y)
            y = y + dy

        print(np.linalg.norm(dy))

        return x, y
    
    def solve_iteration(self, x, y):

        A = self.construct_A()
        F = self.construct_F(x, y)
        J = self.construct_J(x, y)
        
        dy = np.linalg.solve(A - J, F - A @ y.flatten())
        return dy.reshape(4, self.nx)
    
    def construct_A(self):
        nx = self.nx
        def tridiag(a, b, c, k1=-1, k2=0, k3=1):
            return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

        small_diag = np.ones(nx - 1)
        main_diag = np.ones(nx)

        A_left_bc = tridiag(-1/2 * small_diag, 0 * main_diag, 1/2 * small_diag, -1, 0, 1)
        A_right_bc = tridiag(-1/2 * small_diag, 0 * main_diag, 1/2 * small_diag, -1, 0, 1)

        A_left_bc[0, 0] = 1
        A_left_bc[0, 1] = 0
        A_left_bc[-1, -1] = -3/2
        A_left_bc[-1, -2] = 2
        A_left_bc[-1, -3] = -1/2

        A_right_bc[-1, -1] = 1
        A_right_bc[-1, -2] = 0
        A_right_bc[0, 0] = -3/2
        A_right_bc[0, 1] = 2
        A_right_bc[0, 2] = -1/2

        A = np.zeros((4 * nx, 4 * nx))
        matrices = [A_left_bc, A_left_bc, A_right_bc, A_right_bc]
        for i in range(4):
            A[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = matrices[i]

        return A / self.dx
    
    def construct_F(self, x , y):

        F = np.array(self.deriv(x, y))

        # left boundary
        F[0, 0] = 1
        F[1, 0] = 0

        # right boundary
        F[2, -1] = y[1, -1] # / h_x 
        F[3, -1] = - y[0, -1] # / h_x

        F = F.flatten()

        return F
    
    def construct_J(self, x, y):
        y_flat = y.flatten()
        F = self.construct_F(x, y)
        J = np.zeros((len(y_flat), len(y_flat)))
        eps = 1e-7
        for i in range(len(y_flat)):
            y_flat[i] += eps
            F_new = self.construct_F(x, y_flat.reshape(4, self.nx))
            J[:, i] = (F_new - F) / eps
            y_flat[i] -= eps

        return J
    
    
    

    
counter = 0
class CSWEHarm_own():

    def __init__(self):
        
        self.A = 0.72
        self.H = 7.12 
        self.L = 8e3
        

        self.g = 9.81
        self.sigma = 1.4e-4
        self.rho_w = 1025
        self.rho_s = 2650

        self.h0 = 0.0025
        p = 0.4
        self.C_D = 0.0025
        self.lmbda = 6.8e-6
        self.d50 = 0.13e-3
        self.kappa = self.g * self.H / (self.sigma * self.L) ** 2

        print(self.kappa)

        self.a_r = 1e-2
        self.r = 0.24

        self.epsilon = self.A / self.H
        self.eta = self.sigma * self.L / sqrt(self.g * self.H)
        self.U = self.epsilon * self.sigma * self.L

        self.small_number = 1e-5
        self.threshold = 0.995

    def __generate_solution(self):
        # for plotting: store numerical values
        self.t = np.linspace(0, 2 * pi, 1000)
        self.u0_xt = 0 + \
            self.u0_c[:, np.newaxis] * cos(self.t[np.newaxis, :]) + \
            self.u0_s[:, np.newaxis] * sin(self.t[np.newaxis, :])
        self.dz0_xt = 0 + \
            self.dz0_c[:, np.newaxis] * cos(self.t[np.newaxis, :]) + \
            self.dz0_s[:, np.newaxis] * sin(self.t[np.newaxis, :])
        
        self.dz1_xt = 0 + \
            np.tile(self.dz1_r, (len(self.t), 1)).T + \
            self.dz1_c[:, np.newaxis] * cos(2 * self.t[np.newaxis, :]) + \
            self.dz1_s[:, np.newaxis] * sin(2 * self.t[np.newaxis, :])
        
        self.u1_xt = 0 + \
            np.tile(self.u1_r, (len(self.t), 1)).T + \
            self.u1_c[:, np.newaxis] * cos(2 * self.t[np.newaxis, :]) + \
            self.u1_s[:, np.newaxis] * sin(2 * self.t[np.newaxis, :])
        
        
        self.dz_xt = self.dz0_xt + self.epsilon * self.dz1_xt
        self.u_xt  = self.u0_xt  + self.epsilon * self.u1_xt

        return CSWEHarmSolution(self.t, self.x, self.dz0_xt, self.u0_xt, self.dz1_xt, self.u1_xt, self.dz_xt, self.u_xt)

    def solve(self):
        self.solve_LO()
        self.solve_FO()

    def h(self, x):
        return x
        # return 0.9*x

    def h_x(self, x):
        return 1
        # return 0.9
    
    def h_xx(self, x):
        return 0
    
    def solve_LO(self):

        def deriv(x, y):
            dz_c, dz_s, u_c, u_s = y

            h = self.h(x)
            h_x = self.h_x(x)
            friction_term = - self.r / (1 - h + self.h0)

            ddz_c = 1 / self.kappa * (friction_term * u_c - u_s)
            ddz_s = 1 / self.kappa * (friction_term * u_s + u_c)

            du_c, du_s = zeros(u_c.shape), zeros(u_s.shape)

            du_c = (-dz_s + u_c * h_x)/ (1 - h)
            du_s = ( dz_c + u_s * h_x)  / (1 - h) 
            return [ddz_c, ddz_s, du_c, du_s]
        
        my_LO = LO_own(deriv)
        x, y = my_LO.solve()
         
        self.dz0_c, self.dz0_s, self.u0_c, self.u0_s = y

        print(x.shape, y.shape)

        tck = scipy.interpolate.splrep(x, y.T, s=0, k=3) # s=0 for no smoothing, k=3 for cubic
        pp = scipy.interpolate.PPoly.from_spline(tck)

        self.pol0 = pp

    def solve_FO(self):

        def deriv_old(x, vector):
            dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = vector

            dz0_c    , dz0_s    , u0_c    , u0_s     = self.pol0(x)
            dz0_c_dx , dz0_s_dx , u0_c_dx , u0_s_dx  = self.pol0(x, nu=1)
            dz0_c_dxx, dz0_s_dxx, u0_c_dxx, u0_s_dxx = self.pol0(x, nu=2)
            

            # print(dz0_c   , dz0_s   , u0_c   , u0_s, dz0_c_dx, dz0_s_dx, u0_c_dx, u0_s_dx )

            h, h_x, h_xx = self.h(x), self.h_x(x), self.h_xx(x)
            
            # where x < 0.9
            # if x > 0.9, change equations: because 1 - h approx infinity, the factor multiplying it should be 0`

            # continuity equations
            du1_r = 1 / (1 - h + self.small_number) * (
                h_x * u1_r             - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            )
            du1_c = 1 / (1 - h + self.small_number) * (
                h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) # (-dz0_s * u0_s +  dz0_c * u0_c)
            ) 
            du1_s = 1 / (1 - h + self.small_number) * (
                h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s) # ( dz0_s * u0_c +  dz0_c * u0_s)
            ) 





            # momentum equations
            ddz1_r = (1 / (1 - h + self.h0) * (
                - self.r * u1_r
                - 1 / 2 * (  dz0_c *  u0_s - dz0_s *  u0_c)
                - 1 / 2 * (  dz0_s * dz0_s_dx + dz0_c * dz0_c_dx) * self.kappa
            ) - 1/2 * (u0_c * u0_c_dx + u0_s * u0_s_dx)) / self.kappa
            ddz1_c = (1 / (1 - h + self.h0) * (
                - self.r * u1_c
                - 1 / 2 * (  dz0_c *  u0_s + dz0_s *  u0_c)
                - 1 / 2 * (- dz0_s * dz0_s_dx + dz0_c * dz0_c_dx) * self.kappa
            ) - 1/2 * (u0_c * u0_c_dx - u0_s * u0_s_dx) - 2 * u1_s) / self.kappa
            ddz1_s = (1 / (1 - h + self.h0) * (
                - self.r * u1_s
                + 1 / 2 * (  dz0_c *  u0_c - dz0_s *  u0_s)
                - 1 / 2 * (  dz0_c * dz0_s_dx + dz0_s * dz0_c_dx) * self.kappa
            ) - 1/2 * (u0_c * u0_s_dx + u0_s * u0_c_dx) + 2 * u1_c) / self.kappa


            # for elem in [ddz1_r, ddz1_c, ddz1_s, du1_r, du1_c, du1_s]:
            #     print(elem.shape, min(elem), max(elem), min(abs(elem)))
            # print(" ")
                # for val in elem:
                #     print(val, end='\n')
                # print(('--' * 50 + "\n")  * 20)

            
            cc_xx = (dz0_c * u0_c_dxx + 2 * dz0_c_dx * u0_c_dx + dz0_c_dxx * u0_c)
            ss_xx = (dz0_s * u0_s_dxx + 2 * dz0_s_dx * u0_s_dx + dz0_s_dxx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            cs_xx = (dz0_s * u0_c_dxx + 2 * dz0_s_dx * u0_c_dx + dz0_s_dxx * u0_c) # ( dz0_s * u0_s +  dz0_c * u0_c)
            sc_xx = (dz0_c * u0_s_dxx + 2 * dz0_c_dx * u0_s_dx + dz0_c_dxx * u0_s)
                        
            du1_r_hop = (        - u1_c * h_xx + (cc_xx + ss_xx) / 2) / (2*h_x)
            du1_c_hop = ( ddz1_s - u1_c * h_xx + (cc_xx - ss_xx) / 2) / (2*h_x)
            du1_s_hop = (-ddz1_c - u1_s * h_xx + (cs_xx + sc_xx) / 2) / (2*h_x)

            du1_r = np.where(h < self.threshold, du1_r, du1_r_hop)
            du1_c = np.where(h < self.threshold, du1_c, du1_c_hop)
            du1_s = np.where(h < self.threshold, du1_s, du1_s_hop)



                



            num_r = h_x * u1_r             - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            num_c = h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) # (-dz0_s * u0_s +  dz0_c * u0_c)
            num_s = h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s) # ( dz0_s * u0_c +  dz0_c * u0_s)
            print(num_r[-4:])
            print(num_c[-4:])
            print(num_s[-4:])
            print('-' * 5)

            return [ddz1_r, ddz1_c, ddz1_s, du1_r, du1_c, du1_s]
        
        def deriv_old2(x, vector):
            dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = vector

            dz0_c    , dz0_s    , u0_c    , u0_s     = self.pol0(x)
            dz0_c_dx , dz0_s_dx , u0_c_dx , u0_s_dx  = self.pol0(x, nu=1)
            dz0_c_dxx, dz0_s_dxx, u0_c_dxx, u0_s_dxx = self.pol0(x, nu=2)
            

            # print(dz0_c   , dz0_s   , u0_c   , u0_s, dz0_c_dx, dz0_s_dx, u0_c_dx, u0_s_dx )

            h, h_x, h_xx = self.h(x), self.h_x(x), self.h_xx(x)
            
            # where x < 0.9
            # if x > 0.9, change equations: because 1 - h approx infinity, the factor multiplying it should be 0`

            
            # momentum equations
            ddz1_r = (1 / (1 - h + self.h0) * (
                - self.r * u1_r
                - 1 / 2 * (  dz0_c *  u0_s - dz0_s *  u0_c)
                - 1 / 2 * (  dz0_s * dz0_s_dx + dz0_c * dz0_c_dx) * self.kappa
            ) - 1/2 * (u0_c * u0_c_dx + u0_s * u0_s_dx)) / self.kappa
            ddz1_c = (1 / (1 - h + self.h0) * (
                - self.r * u1_c
                - 1 / 2 * (  dz0_c *  u0_s + dz0_s *  u0_c)
                - 1 / 2 * (- dz0_s * dz0_s_dx + dz0_c * dz0_c_dx) * self.kappa
            ) - 1/2 * (u0_c * u0_c_dx - u0_s * u0_s_dx) - 2 * u1_s) / self.kappa
            ddz1_s = (1 / (1 - h + self.h0) * (
                - self.r * u1_s
                + 1 / 2 * (  dz0_c *  u0_c - dz0_s *  u0_s)
                - 1 / 2 * (  dz0_c * dz0_s_dx + dz0_s * dz0_c_dx) * self.kappa
            ) - 1/2 * (u0_c * u0_s_dx + u0_s * u0_c_dx) + 2 * u1_c) / self.kappa

            # continuity equations
            boundary = -1
            du1_r = zeros(u1_r.shape)
            du1_c = zeros(u1_c.shape)
            du1_s = zeros(u1_s.shape)

            du1_r[:boundary] = 1 / (1 - h)[:boundary] * (
                h_x * u1_r             - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            )[:boundary]
            du1_c[:boundary] = 1 / (1 - h)[:boundary] * (
                h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) # (-dz0_s * u0_s +  dz0_c * u0_c)
            )[:boundary]
            du1_s[:boundary] = 1 / (1 - h)[:boundary] * (
                h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s) # ( dz0_s * u0_c +  dz0_c * u0_s)
            )[:boundary]

            cc_xx = (dz0_c * u0_c_dxx + 2 * dz0_c_dx * u0_c_dx + dz0_c_dxx * u0_c)
            ss_xx = (dz0_s * u0_s_dxx + 2 * dz0_s_dx * u0_s_dx + dz0_s_dxx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            cs_xx = (dz0_s * u0_c_dxx + 2 * dz0_s_dx * u0_c_dx + dz0_s_dxx * u0_c) # ( dz0_s * u0_s +  dz0_c * u0_c)
            sc_xx = (dz0_c * u0_s_dxx + 2 * dz0_c_dx * u0_s_dx + dz0_c_dxx * u0_s)
                    

            du1_r[boundary:] = (        - u1_c * h_xx + (cc_xx + ss_xx) / 2)[boundary:] / (2*h_x)
            du1_c[boundary:] = ( ddz1_s - u1_c * h_xx + (cc_xx - ss_xx) / 2)[boundary:] / (2*h_x)
            du1_s[boundary:] = (-ddz1_c - u1_s * h_xx + (cs_xx + sc_xx) / 2)[boundary:] / (2*h_x)



            # for elem in [ddz1_r, ddz1_c, ddz1_s, du1_r, du1_c, du1_s]:
            #     print(elem.shape, min(elem), max(elem), min(abs(elem)))
            # print(" ")
                # for val in elem:
                #     print(val, end='\n')
                # print(('--' * 50 + "\n")  * 20)


            num_r = h_x * u1_r             - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            num_c = h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) # (-dz0_s * u0_s +  dz0_c * u0_c)
            num_s = h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s) # ( dz0_s * u0_c +  dz0_c * u0_s)
            # print(num_r[-4:])
            # print(num_c[-4:])
            # print(num_s[-4:])
            # print('-' * 5)

            return [ddz1_r, ddz1_c, ddz1_s, du1_r, du1_c, du1_s]
        
        def deriv(x, vector):
            dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = vector

            dz0_c    , dz0_s    , u0_c    , u0_s     = self.pol0(x)
            dz0_c_dx , dz0_s_dx , u0_c_dx , u0_s_dx  = self.pol0(x, nu=1)
            dz0_c_dxx, dz0_s_dxx, u0_c_dxx, u0_s_dxx = self.pol0(x, nu=2)
            
            h, h_x, h_xx = self.h(x), self.h_x(x), self.h_xx(x)
            
            # momentum equations
            ddz1_r = (1 / (1 - h + self.h0) * (
                - self.r * u1_r
                - 1 / 2 * (  dz0_c *  u0_s - dz0_s *  u0_c)
                - 1 / 2 * (  dz0_s * dz0_s_dx + dz0_c * dz0_c_dx) * self.kappa
            ) - 1/2 * (u0_c * u0_c_dx + u0_s * u0_s_dx)) / self.kappa
            ddz1_c = (1 / (1 - h + self.h0) * (
                - self.r * u1_c
                - 1 / 2 * (  dz0_c *  u0_s + dz0_s *  u0_c)
                - 1 / 2 * (- dz0_s * dz0_s_dx + dz0_c * dz0_c_dx) * self.kappa
            ) - 1/2 * (u0_c * u0_c_dx - u0_s * u0_s_dx) - 2 * u1_s) / self.kappa
            ddz1_s = (1 / (1 - h + self.h0) * (
                - self.r * u1_s
                + 1 / 2 * (  dz0_c *  u0_c - dz0_s *  u0_s)
                - 1 / 2 * (  dz0_c * dz0_s_dx + dz0_s * dz0_c_dx) * self.kappa
            ) - 1/2 * (u0_c * u0_s_dx + u0_s * u0_c_dx) + 2 * u1_c) / self.kappa


            du1_r = 1 / (1 - h)* (
                h_x * u1_r             - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            )
            du1_c = 1 / (1 - h) * (
                h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) # (-dz0_s * u0_s +  dz0_c * u0_c)
            )
            du1_s = 1 / (1 - h) * (
                h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s) # ( dz0_s * u0_c +  dz0_c * u0_s)
            )

            a1 = h_x * u1_r - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            a2 = h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) # (-dz0_s * u0_s +  dz0_c * u0_c)
            a3 = h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s* u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s) # ( dz0_s * u0_c +  dz0_c * u0_s)
            print(a1[-3:], a2[-3:], a3[-3:])


            return [ddz1_r, ddz1_c, ddz1_s, du1_r, du1_c, du1_s]
    
        def bc(vector_left, vector_right):
            dz1_r0, dz1_c0, dz1_s0, u1_r0, u1_c0, u1_s0 = vector_left
            dz1_r1, dz1_c1, dz1_s1, u1_r1, u1_c1, u1_s1 = vector_right

            # only at the right boundary, the leading order solution gets used
            dz0_c   , dz0_s   , u0_c   , u0_s    = self.pol0(1)
            dz0_c_dx, dz0_s_dx, u0_c_dx, u0_s_dx = self.pol0(1, nu=1)

            h_x = self.h_x(1)
            a1 = h_x * u1_r1 - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            a2 = h_x * u1_c1 - 2 * dz1_s1 - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) # (-dz0_s * u0_s +  dz0_c * u0_c)
            a3 = h_x * u1_s1 + 2 * dz1_c1 - 1 / 2 * (dz0_s* u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s) # ( dz0_s * u0_c +  dz0_c * u0_s)
            print(a1, a2, a3)
            return [
                dz1_r0, dz1_s0, dz1_c0,
                h_x * u1_r1 - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s),
                h_x * u1_c1 - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) - 2 * dz1_s1,
                h_x * u1_s1 - 1 / 2 * (dz0_c * u0_s_dx + dz0_c_dx * u0_s + dz0_s * u0_c_dx + dz0_s_dx * u0_c) + 2 * dz1_c1
            ]
        
        vector_guess = 0.1 * np.ones((6, len(self.x)))

        sol = scipy.integrate.solve_bvp(deriv, bc, self.x, vector_guess, tol=1e-3, max_nodes=20000)
        print(sol)

        if sol.status:
            raise SystemError

        self.x1 = sol.x
        self.dz1_r, self.dz1_c, self.dz1_s, self.u1_r, self.u1_c, self.u1_s = sol.y
        self.pol1 = sol.sol

        return sol

    def plot_components(self):
        harmonic_components(self.x0, zeros(self.x0.shape), self.dz0_c, self.dz0_s, "LO $\\zeta$")
        harmonic_components(self.x1, self.dz1_r,          self.dz1_c, self.dz1_s, "FO $\\zeta$")
        harmonic_components(self.x0, zeros(self.x0.shape), self.u0_c,  self.u0_s,  "LO u")
        harmonic_components(self.x1, self.u1_r,           self.u1_c,  self.u1_s,  "FO u")

    def generate_meshes(self):
        self.t = np.linspace(0, 2*pi, 1000)
        self.x = np.linspace(0, 1, 1000)

        dz0_c, dz0_s, u0_c, u0_s = self.pol0(self.x)
        dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = self.pol1(self.x)

        self.dz0_xt = 0 + \
            dz0_c[:, np.newaxis] * cos(self.t[np.newaxis, :]) + \
            dz0_s[:, np.newaxis] * sin(self.t[np.newaxis, :])
        

        self.u0_xt = 0 + \
            u0_c[:, np.newaxis] * cos(self.t[np.newaxis, :]) + \
            u0_s[:, np.newaxis] * sin(self.t[np.newaxis, :])
        
        self.dz1_xt = 0 + \
            np.tile(dz1_r, (len(self.t), 1)).T + \
            dz1_c[:, np.newaxis] * cos(2 * self.t[np.newaxis, :]) + \
            dz1_s[:, np.newaxis] * sin(2 * self.t[np.newaxis, :])
        
        self.u1_xt = 0 + \
            np.tile(u1_r, (len(self.t), 1)).T + \
            u1_c[:, np.newaxis] * cos(2 * self.t[np.newaxis, :]) + \
            u1_s[:, np.newaxis] * sin(2 * self.t[np.newaxis, :])
        
        self.dz_xt = self.dz0_xt + self.epsilon * self.dz1_xt
        self.u_xt  = self.u0_xt  + self.epsilon * self.u1_xt
        
        self.T_mesh, self.X_mesh = np.meshgrid(self.t, self.x)        

    def heatmaps(self):

        heatmap(self.T_mesh, self.X_mesh, self.dz0_xt, "dzeta0(x,t)")
        heatmap(self.T_mesh, self.X_mesh, self.u0_xt, "u0(x,t)")

        heatmap(self.T_mesh, self.X_mesh, self.dz1_xt, "dzeta1(x,t)")
        heatmap(self.T_mesh, self.X_mesh, self.u1_xt, "u1(x,t)")

        heatmap(self.T_mesh, self.X_mesh, self.dz_xt, "dzeta(x,t)")
        heatmap(self.T_mesh, self.X_mesh, self.u_xt, "u(x,t)")





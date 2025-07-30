

import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt, zeros, ones
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm

from plots import *

from cswe_harm_solution import CSWEHarmSolution

counter = 0
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
        p = 0.4
        self.C_D = 0.0025
        self.lmbda = 6.8e-6
        self.d50 = 0.13e-3
        self.kappa = self.g * self.H / (self.sigma * self.L) ** 2

        self.a_r = 1e-2
        self.r = 0.24

        self.epsilon = self.A / self.H
        self.eta = self.sigma * self.L / sqrt(self.g * self.H)
        self.U = self.epsilon * self.sigma * self.L

        self.small_number = 1e-5
        self.domain_reduction = 1e-5

        self.boundary = 0.99


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
    
    def solve_LO_small_number(self):
        def deriv(x, y):
            dz_c, dz_s, u_c, u_s = y

            h = self.h(x)
            h_x = self.h_x(x)
            friction_term = - self.r / (1 - h + self.h0)

            ddz_c = 1 / self.kappa * (friction_term * u_c - u_s)
            ddz_s = 1 / self.kappa * (friction_term * u_s + u_c)

            du_c, du_s = zeros(u_c.shape), zeros(u_s.shape)

            du_c = (-dz_s + u_c * h_x)/ (1 - h + self.small_number)
            du_s = ( dz_c + u_s * h_x)  / (1 - h + self.small_number) 
            return [ddz_c, ddz_s, du_c, du_s]
        
        def bc(y_left, y_right):
            dz_c0, dz_s0, u_c0, u_s0 = y_left
            dz_c1, dz_s1, u_c1, u_s1 = y_right

            h_x = self.h_x(1)

            return [
                dz_c0 - 1,
                dz_s0,
                dz_s1 - h_x * u_c1,
                dz_c1 + h_x * u_s1
            ]


        self.x = linspace(0, 1, 1000)
        self.y_guess = np.zeros((4, len(self.x)))
        self.y0 = scipy.integrate.solve_bvp(deriv, bc, self.x, self.y_guess, tol=1e-6, max_nodes=20000)

    def solve_LO_reduced_domain(self):
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
        
        def bc(y_left, y_right):
            dz_c0, dz_s0, u_c0, u_s0 = y_left
            dz_c1, dz_s1, u_c1, u_s1 = y_right

            h_x = self.h_x(1)

            return [
                dz_c0 - 1,
                dz_s0,
                dz_s1 - h_x * u_c1,
                dz_c1 + h_x * u_s1
            ]


        self.x = linspace(0, 1 - self.domain_reduction, 1000)
        self.y_guess = np.zeros((4, len(self.x)))
        self.y0 = scipy.integrate.solve_bvp(deriv, bc, self.x, self.y_guess, tol=1e-6, max_nodes=25000)

    def solve_LO_split_domain(self):

        def deriv_l(x, y):
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
        
        def deriv_r(x, y):
            dz_c, dz_s, u_c, u_s = y

            h = self.h(x)
            h_x = self.h_x(x)
            h_xx = self.h_xx(x)
            friction_term = - self.r / (1 - h + self.h0)

            ddz_c = 1 / self.kappa * (friction_term * u_c - u_s)
            ddz_s = 1 / self.kappa * (friction_term * u_s + u_c)

            du_c, du_s = zeros(u_c.shape), zeros(u_s.shape)

            du_c =  ( ddz_s - u_c * h_xx)  / (2*h_x)
            du_s =  (-ddz_c - u_s * h_xx)  / (2*h_x)
            return [ddz_c, ddz_s, du_c, du_s]
        
        def bc_l(y_left, y_right):
            dz_c0, dz_s0, u_c0, u_s0 = y_left
            dz_c1, dz_s1, u_c1, u_s1 = y_right

            h_x = self.h_x(1)

            return [
                dz_c0 - 1,
                dz_s0,
                dz_c1 - self.p[0],
                dz_s1 - self.p[1]
            ]
        
        def bc_r(y_left, y_right):
            dz_c0, dz_s0, u_c0, u_s0 = y_left
            dz_c1, dz_s1, u_c1, u_s1 = y_right

            h_x = self.h_x(1)

            return [
                dz_c0 - self.p[0],
                dz_s0 - self.p[1],
                dz_s1 - h_x * u_c1,
                dz_c1 + h_x * u_s1
            ]
        
        def generate_u_diff(plot=True):

            
            self.yl0 = scipy.integrate.solve_bvp(deriv_l, bc_l, self.xl, self.yl_guess, tol=1e-6, max_nodes=20000)
            if self.debug and 0:
                print(self.yl0)
            self.yr0 = scipy.integrate.solve_bvp(deriv_r, bc_r, self.xr, self.yr_guess, tol=1e-6, max_nodes=20000)
            if self.debug and 0:
                print(self.yr0)
            if self.yl0.status != 0 or self.yr0.status != 0:
                print(self.yl0)
                print(self.yr0)
                raise SystemError
            

            if self.debug:
                if plot:
                    st = int(len(self.yl0.x) * (1 - (1 - self.boundary) * 10 / self.boundary))
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                    for i in range(4):
                        axs[i].plot(self.yl0.x[st:], self.yl0.y[i, st:])
                        axs[i].plot(self.yr0.x, self.yr0.y[i])
                    plt.show()

            u_l = self.yl0.y[2:, -1] 
            u_r = self.yr0.y[2:, 0]

            u_diff = u_r - u_l
            return u_diff
        
        def generate_J_u_diff():

            u_diff = generate_u_diff(plot=False)
            eps = 1e-7
            J = zeros((len(self.p), len(self.p)))
            for i in range(len(self.p)):
                self.p[i] += eps
                new_u_diff = generate_u_diff(plot=False)
                J[:, i] = (new_u_diff - u_diff) / eps
                self.p[i] -= eps

            # test J_u
            if self.debug:
                test_p = np.random.randn(2) * 1e-2
                print('test')
                self.p += test_p
                print(generate_u_diff(plot=False))
                self.p -= test_p
                print(u_diff + J @ test_p)
                print(u_diff)

            return J


        self.xl = linspace(0, self.boundary, 10000)
        self.xr = linspace(self.boundary, 1, 10000)

        s = 0.1
        self.yl_guess = s * np.ones((4, len(self.xl)))
        self.yr_guess = s * np.ones((4, len(self.xr)))
        self.p = np.array([s, s])

        u_diff = generate_u_diff()
        counter = 0
        while np.linalg.norm(u_diff) > 1e-12 and counter < 10:
            print(np.linalg.norm(u_diff))
            counter += 1
            
            J_u_diff = generate_J_u_diff()

            ideal_boundary = np.linalg.solve(J_u_diff, u_diff)
            if self.debug:
                print("here")
                print(self.p)
                print(u_diff)
                print(J_u_diff)
                print(ideal_boundary)

            self.p = self.p - ideal_boundary
            u_diff = generate_u_diff()


        print(np.linalg.norm(u_diff))
        return

    def solve_FO_split_domain(self):


        def deriv_l1(x, y):
            dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = y

            dz0_c    , dz0_s    , u0_c    , u0_s     = self.yl0.sol(x)
            dz0_c_dx , dz0_s_dx , u0_c_dx , u0_s_dx  = self.yl0.sol(x, nu=1)
            dz0_c_dxx, dz0_s_dxx, u0_c_dxx, u0_s_dxx = self.yl0.sol(x, nu=2)

            # if self.debug:
            #     plt.plot(x, dz0_c)
            #     plt.plot(x, dz0_c_dx)
            #     plt.plot(x, dz0_s)
            #     plt.plot(x, u0_c)
            #     plt.plot(x, u0_s)



            h, h_x, h_xx = self.h(x), self.h_x(x), self.h_xx(x)
            
            # continuity equations
            du1_r = 1 / (1 - h) * (h_x * u1_r             - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s))
            du1_c = 1 / (1 - h) * (h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s))
            du1_s = 1 / (1 - h) * (h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s))
            
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
        
            return [ddz1_r, ddz1_c, ddz1_s, du1_r, du1_c, du1_s]
        
        def deriv_r1(x, y):
            dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = y

            dz0_c    , dz0_s    , u0_c    , u0_s     = self.yr0.sol(x)
            dz0_c_dx , dz0_s_dx , u0_c_dx , u0_s_dx  = self.yr0.sol(x, nu=1)
            dz0_c_dxx, dz0_s_dxx, u0_c_dxx, u0_s_dxx = self.yr0.sol(x, nu=2)

            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            for i in range(4): axs[i].plot(x, self.yr0.sol(x)[i])
            plt.show()

            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            for i in range(4): axs[i].plot(x, self.yr0.sol(x, nu = 1)[i])
            plt.show()

            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            for i in range(4): axs[i].plot(x, self.yr0.sol(x, nu = 2)[i])
            plt.show()

            raise SystemError


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
            
            cc_xx = (dz0_c * u0_c_dxx + 2 * dz0_c_dx * u0_c_dx + dz0_c_dxx * u0_c)
            ss_xx = (dz0_s * u0_s_dxx + 2 * dz0_s_dx * u0_s_dx + dz0_s_dxx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            cs_xx = (dz0_s * u0_c_dxx + 2 * dz0_s_dx * u0_c_dx + dz0_s_dxx * u0_c) # ( dz0_s * u0_s +  dz0_c * u0_c)
            sc_xx = (dz0_c * u0_s_dxx + 2 * dz0_c_dx * u0_s_dx + dz0_c_dxx * u0_s)
                        
            du1_r_hop = (          - u1_c * h_xx + (cc_xx + ss_xx) / 2) / (2*h_x)
            du1_c_hop = ( 2*ddz1_s - u1_c * h_xx + (cc_xx - ss_xx) / 2) / (2*h_x)
            du1_s_hop = (-2*ddz1_c - u1_s * h_xx + (cs_xx + sc_xx) / 2) / (2*h_x)
            

            return [ddz1_r, ddz1_c, ddz1_s, du1_r_hop, du1_c_hop, du1_s_hop]
        
        def bc_l1(y_left, y_right):
            dz1_r0, dz1_c0, dz1_s0, u1_r0, u1_c0, u1_s0 = y_left
            dz1_r1, dz1_c1, dz1_s1, u1_r1, u1_c1, u1_s1 = y_right

            return [
                dz1_r0, dz1_s0, dz1_c0,
                dz1_r1 - self.p[0],
                dz1_c1 - self.p[1],
                dz1_s1 - self.p[2]
            ]
        
        def bc_r1(y_left, y_right):
            dz1_r0, dz1_c0, dz1_s0, u1_r0, u1_c0, u1_s0 = y_left
            dz1_r1, dz1_c1, dz1_s1, u1_r1, u1_c1, u1_s1 = y_right

            # only at the right boundary, the leading order solution gets used
            dz0_c   , dz0_s   , u0_c   , u0_s    = self.yr0.sol(1)
            dz0_c_dx, dz0_s_dx, u0_c_dx, u0_s_dx = self.yr0.sol(1, nu=1)

            h_x = self.h_x(1)
            return [
                dz1_r0 - self.p[0],
                dz1_c0 - self.p[1],
                dz1_s0 - self.p[2],
                h_x * u1_r1 - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s),
                h_x * u1_c1 - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) - 2 * dz1_s1,
                h_x * u1_s1 - 1 / 2 * (dz0_c * u0_s_dx + dz0_c_dx * u0_s + dz0_s * u0_c_dx + dz0_s_dx * u0_c) + 2 * dz1_c1
            ]
        
        def generate_u1_diff(plot=True):

            self.yl1 = scipy.integrate.solve_bvp(deriv_l1, bc_l1, self.xl, self.yl_guess, tol=1e-4, max_nodes=5000)
            if self.debug:
                print(self.yl1)
            self.yr1 = scipy.integrate.solve_bvp(deriv_r1, bc_r1, self.xr, self.yr_guess, tol=1e-4, max_nodes=5000)
            if self.debug:
                print(self.yr1)

            if self.yl1.status != 0 or self.yr1.status != 0:
                print(self.yl1)
                print(self.yr1)
                raise SystemError
            

            if self.debug:
                if plot:
                    st = len(self.yl1.x) - int((1 - self.boundary) * 10 / self.boundary * len(self.yl1.x))
                    fig, axs = plt.subplots(1, 6, figsize=(30, 5))

                    for i in range(6):
                        axs[i].plot(self.yl1.x[st:], self.yl1.y[i, st:])
                        axs[i].plot(self.yr1.x, self.yr1.y[i])
                    plt.show()
                    print("-\n" * 10)

            u_l = self.yl1.y[3:, -1] 
            u_r = self.yr1.y[3:, 0]

            u_diff = u_r - u_l
            return u_diff

        def generate_J_u1_diff():

            u_diff = generate_u1_diff(plot=False)
            eps = 1e-7
            J = zeros((len(self.p), len(self.p)))
            for i in range(len(self.p)):
                self.p[i] += eps
                new_u_diff = generate_u1_diff(plot=False)
                J[:, i] = (new_u_diff - u_diff) / eps
                self.p[i] -= eps

            # test J_u
            if self.debug:
                test_p = np.random.randn(3) * 1e-2
                print('test')
                self.p += test_p
                print(generate_u1_diff(plot=False))
                self.p -= test_p
                print(u_diff + J @ test_p)
                print(u_diff)

            return J
        


        self.xl = linspace(0, self.boundary, 1000)
        self.xr = linspace(self.boundary, 1, 1000)

        s = 1e-3
        self.yl_guess = s * np.ones((6, len(self.xl)))
        self.yr_guess = s * np.ones((6, len(self.xr)))
        self.p = np.array([s, s, s])

        u_diff = generate_u1_diff()
        counter = 0
        while np.linalg.norm(u_diff) > 1e-10 and counter < 6:
            print(u_diff)
            print(np.linalg.norm(u_diff))
            counter += 1
            
            J_u_diff = generate_J_u1_diff()

            ideal_boundary = np.linalg.solve(J_u_diff, u_diff)
            if self.debug:
                print("here")
                print(self.p)
                print(u_diff)
                print(J_u_diff)
                print(ideal_boundary)

            self.p = self.p - ideal_boundary
            u_diff = generate_u1_diff()


        print(np.linalg.norm(u_diff))
        return
     
    def solve_FO_reduced_domain(self):

    
        def deriv(x, vector):
            dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = vector

            dz0_c    , dz0_s    , u0_c    , u0_s     = self.y0.sol(x)
            dz0_c_dx , dz0_s_dx , u0_c_dx , u0_s_dx  = self.y0.sol(x, nu=1)
            dz0_c_dxx, dz0_s_dxx, u0_c_dxx, u0_s_dxx = self.y0.sol(x, nu=2)
            
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

            if self.debug:
                a1 = h_x * u1_r - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
                a2 = h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) # (-dz0_s * u0_s +  dz0_c * u0_c)
                a3 = h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s* u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s) # ( dz0_s * u0_c +  dz0_c * u0_s)
                print(a1[-3:], a2[-3:], a3[-3:])

            return [ddz1_r, ddz1_c, ddz1_s, du1_r, du1_c, du1_s]
    
        def bc(vector_left, vector_right):
            dz1_r0, dz1_c0, dz1_s0, u1_r0, u1_c0, u1_s0 = vector_left
            dz1_r1, dz1_c1, dz1_s1, u1_r1, u1_c1, u1_s1 = vector_right

            # only at the right boundary, the leading order solution gets used
            dz0_c   , dz0_s   , u0_c   , u0_s    = self.y0.sol(1)
            dz0_c_dx, dz0_s_dx, u0_c_dx, u0_s_dx = self.y0.sol(1, nu=1)

            h_x = self.h_x(1)
            a1 = h_x * u1_r1 - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
            a2 = h_x * u1_c1 - 2 * dz1_s1 - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) # (-dz0_s * u0_s +  dz0_c * u0_c)
            a3 = h_x * u1_s1 + 2 * dz1_c1 - 1 / 2 * (dz0_s* u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s) # ( dz0_s * u0_c +  dz0_c * u0_s)
            if self.debug:
                print(a1, a2, a3)
            return [
                dz1_r0, dz1_s0, dz1_c0,
                h_x * u1_r1 - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s),
                h_x * u1_c1 - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) - 2 * dz1_s1,
                h_x * u1_s1 - 1 / 2 * (dz0_c * u0_s_dx + dz0_c_dx * u0_s + dz0_s * u0_c_dx + dz0_s_dx * u0_c) + 2 * dz1_c1
            ]
        
        vector_guess = 0.1 * np.ones((6, len(self.x)))

        sol = scipy.integrate.solve_bvp(deriv, bc, self.x, vector_guess, tol=1e-6, max_nodes=20000)
        if self.debug:
            print(sol)

        if sol.status:
            print(sol)
            raise SystemError
        

        self.y1 = sol 


        # outdated stuff
        self.x1 = sol.x
        self.dz1_r, self.dz1_c, self.dz1_s, self.u1_r, self.u1_c, self.u1_s = sol.y
        self.pol1 = sol.sol

        return sol
    
    def test_FO(self):

        test_x = linspace(0, 1, 5000)


 

        dz0_c = scipy.interpolate.PPoly(self.pol0.c[:, :, 0], self.pol0.x)
        dz0_s = scipy.interpolate.PPoly(self.pol0.c[:, :, 1], self.pol0.x)
        u0_c = scipy.interpolate.PPoly(self.pol0.c[:, :, 2], self.pol0.x)
        u0_s = scipy.interpolate.PPoly(self.pol0.c[:, :, 3], self.pol0.x)

        dz1_r = scipy.interpolate.PPoly(self.pol1.c[:, :, 0], self.pol1.x)
        dz1_c = scipy.interpolate.PPoly(self.pol1.c[:, :, 1], self.pol1.x)
        dz1_s = scipy.interpolate.PPoly(self.pol1.c[:, :, 2], self.pol1.x)
        u1_r = scipy.interpolate.PPoly(self.pol1.c[:, :, 3], self.pol1.x)
        u1_c = scipy.interpolate.PPoly(self.pol1.c[:, :, 4], self.pol1.x)
        u1_s = scipy.interpolate.PPoly(self.pol1.c[:, :, 5], self.pol1.x)

        h_x = self.h_x(test_x)
        dz0_c_dx = dz0_c(test_x, nu = 1)
        dz0_s_dx = dz0_s(test_x, nu = 1)
        u0_c_dx = u0_c(test_x, nu = 1)
        u0_s_dx = dz0_s(test_x, nu = 1)
        a1 = h_x * u1_r(test_x) - 1 / 2 * (dz0_c(test_x) * u0_c_dx + dz0_c_dx * u0_c(test_x) + dz0_s(test_x) * u0_s_dx + dz0_s_dx * u0_s(test_x)) # ( dz0_s * u0_s +  dz0_c * u0_c)
        a2 = h_x * u1_c(test_x) - 2 * dz1_s(test_x) - 1 / 2 * (dz0_c(test_x) * u0_c_dx + dz0_c_dx * u0_c(test_x) - dz0_s(test_x) * u0_s_dx - dz0_s_dx * u0_s(test_x)) # (-dz0_s * u0_s +  dz0_c * u0_c)
        a3 = h_x * u1_s(test_x) + 2 * dz1_c(test_x) - 1 / 2 * (dz0_s(test_x) * u0_c_dx + dz0_s_dx * u0_c(test_x) + dz0_c(test_x) * u0_s_dx + dz0_c_dx * u0_s(test_x)) # ( dz0_s * u0_c +  dz0_c * u0_s)
        plt.plot(test_x, a1)
        plt.plot(test_x, a2)
        plt.plot(test_x, a3)
        plt.show()
        print(a1[-3:], a2[-3:], a3[-3:])


        test_small_number = self.small_number

        # residual con
        comp1 = dz0_s(test_x, nu = 1) * u0_s(test_x) + \
                dz0_s(test_x) * u0_s(test_x, nu = 1) + \
                dz0_c(test_x, nu = 1) * u0_c(test_x) + \
                dz0_c(test_x) * u0_c(test_x, nu = 1)

        comp2 = (1 - self.h(test_x) + test_small_number) * u1_r(test_x, nu = 1)
        comp3 = - self.h_x(test_x) * u1_r(test_x)

        result = comp1 / 2 + comp2 + comp3
        plt.plot(result)
        plt.show()
        print(result)

        # cos con 
        comp1 = -dz0_s(test_x, nu = 1) * u0_s(test_x) + \
                -dz0_s(test_x) * u0_s(test_x, nu = 1) + \
                dz0_c(test_x, nu = 1) * u0_c(test_x) + \
                dz0_c(test_x) * u0_c(test_x, nu = 1)

        comp2 = (1 - self.h(test_x) + test_small_number) * u1_c(test_x, nu = 1)
        comp3 = - self.h_x(test_x) * u1_c(test_x)

        comp4 = 2 * dz1_s(test_x)

        result = comp1 / 2 + comp2 + comp3 + comp4
        plt.plot(result)
        plt.show()
        print(result)

        # sin con
        comp1 = dz0_s(test_x, nu = 1) * u0_c(test_x) + \
                dz0_s(test_x) * u0_c(test_x, nu = 1) + \
                dz0_c(test_x, nu = 1) * u0_s(test_x) + \
                dz0_c(test_x) * u0_s(test_x, nu = 1)

        comp2 = (1 - self.h(test_x) + test_small_number) * u1_s(test_x, nu = 1)
        comp3 = - self.h_x(test_x) * u1_s(test_x)

        comp4 = - 2 * dz1_c(test_x)

        result = comp1 / 2 + comp2 + comp3 + comp4
        plt.plot(result)
        plt.show()
        print(result)


        # residual momentum
        h = self.h(test_x)

        comp1 = u0_c(test_x) * u0_c(test_x, nu = 1) + \
                u0_s(test_x) * u0_s(test_x, nu = 1)
        
        comp2 = self.kappa * dz1_r(test_x, nu = 1)
        comp3 = dz0_c(test_x) * u0_s(test_x) - \
                dz0_s(test_x) * u0_c(test_x)
        
        comp4 = dz0_c(test_x) * dz0_c(test_x, nu = 1) + \
                dz0_s(test_x) * dz0_s(test_x, nu = 1)
        comp5 = self.r * u1_r(test_x)
        result =  (1 - h + self.h0) * (comp1 / 2 + comp2) + comp3 / 2 + comp4 / 2 * self.kappa + comp5
        plt.plot(result)
        plt.show()
        print(result)

        # cos momentum
        h = self.h(test_x)

        comp1 = u0_c(test_x) * u0_c(test_x, nu = 1) - \
                u0_s(test_x) * u0_s(test_x, nu = 1)
        
        comp2 = self.kappa * dz1_c(test_x, nu = 1)
        comp3 = dz0_c(test_x) * u0_s(test_x) + \
                dz0_s(test_x) * u0_c(test_x)
        
        comp4 = dz0_c(test_x) * dz0_c(test_x, nu = 1) - \
                dz0_s(test_x) * dz0_s(test_x, nu = 1)
        comp5 = self.r * u1_c(test_x)

        comp6 = u1_s(test_x)

        result =  (1 - h + self.h0) * (2 * comp6 + comp1 / 2 + comp2) + comp3 / 2 + comp4 / 2 * self.kappa + comp5
        plt.plot(result)
        plt.show()
        print(result)


        # sin momentum
        h = self.h(test_x)

        comp1 = u0_c(test_x) * u0_s(test_x, nu = 1) + \
                u0_s(test_x) * u0_c(test_x, nu = 1)
        
        comp2 = self.kappa * dz1_s(test_x, nu = 1)
        comp3 = dz0_c(test_x) * u0_c(test_x) - \
                dz0_s(test_x) * u0_s(test_x)
        
        comp4 = dz0_c(test_x) * dz0_s(test_x, nu = 1) + \
                dz0_s(test_x) * dz0_c(test_x, nu = 1)
        comp5 = self.r * u1_s(test_x)

        comp6 = u1_c(test_x)

        result =  (1 - h + self.h0) * (-2 * comp6 + comp1 / 2 + comp2) - comp3 / 2 + comp4 / 2 * self.kappa + comp5
        plt.plot(result)
        plt.show()
        print(result)

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

        
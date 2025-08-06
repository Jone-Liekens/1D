


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

    def __init__(self, deriv, upwind = False):
        self.deriv = deriv
        self.nx = 500
        if upwind:
            self.construct_A = self.construct_A_upwind
        else:
            self.construct_A = self.construct_A_central_diff
    
    def solve(self):
        x = np.linspace(0, 1, self.nx)
        self.dx = x[1] - x[0]
        initial_guess = 0.1 * np.ones((4, len(x)))
        
        dy = 1e50 * np.ones(initial_guess.shape)
        y = initial_guess

        while np.linalg.norm(dy) > 1e-12:
            print(np.linalg.norm(dy))
            A = self.construct_A()
            F = self.construct_F(x, y)
            J = self.construct_J(x, y)
            
            dy = np.linalg.solve(A - J, F - A @ y.flatten())
            dy = dy.reshape(4, self.nx)
            y = y + dy

        print(np.linalg.norm(dy))

        return x, y    
    
    def construct_A_central_diff(self):
        nx = self.nx
        def tridiag(a, b, c, k1=-1, k2=0, k3=1):
            return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

        small_diag = np.ones(nx - 1)
        main_diag = np.ones(nx)

        A_left_bc = tridiag(-1/2 * small_diag, 0 * main_diag, 1/2 * small_diag, -1, 0, 1) / self.dx
        A_right_bc = tridiag(-1/2 * small_diag, 0 * main_diag, 1/2 * small_diag, -1, 0, 1) / self.dx

        A_left_bc[0, 0] = 1 
        A_left_bc[0, 1] = 0
        A_left_bc[-1, -1] = -3/2 / self.dx
        A_left_bc[-1, -2] = 2 / self.dx
        A_left_bc[-1, -3] = -1/2 / self.dx

        A_right_bc[-1, -1] = 1
        A_right_bc[-1, -2] = 0
        A_right_bc[0, 0] = -3/2 / self.dx
        A_right_bc[0, 1] = 2 / self.dx
        A_right_bc[0, 2] = -1/2 / self.dx

        A = np.zeros((4 * nx, 4 * nx))
        matrices = [A_left_bc, A_left_bc, A_right_bc, A_right_bc]
        for i in range(4):
            A[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = matrices[i]

        return A
    
    def construct_A_upwind(self):
        nx = self.nx
        def tridiag(a, b, c, k1=-1, k2=0, k3=1):
            return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

        small_diag = np.ones(nx - 1)
        main_diag = np.ones(nx)

        A_left_bc = tridiag(-1 * small_diag, 1 * main_diag, 0 * small_diag, -1, 0, 1) / self.dx
        # A_right_bc = -tridiag(-1 * small_diag, 1 * main_diag, 0 * small_diag, -1, 0, 1) / self.dx

        # A_left_bc = tridiag(0 * small_diag, -1 * main_diag, 1 * small_diag, -1, 0, 1) / self.dx
        A_right_bc = tridiag(0 * small_diag, -1 * main_diag, 1 * small_diag, -1, 0, 1) / self.dx


        A_left_bc[0, 0] = 1 
        A_left_bc[0, 1] = 0
        # A_left_bc[-1, -1] = -3/2 / self.dx
        # A_left_bc[-1, -2] = 2 / self.dx
        # A_left_bc[-1, -3] = -1/2 / self.dx
        A_left_bc[-1, -1] = 1 / self.dx
        A_left_bc[-1, -2] = -1 / self.dx

        A_right_bc[-1, -1] = 1
        A_right_bc[-1, -2] = 0
        # A_right_bc[0, 0] = -3/2 / self.dx
        # A_right_bc[0, 1] = 2 / self.dx
        # A_right_bc[0, 2] = -1/2 / self.dx
        A_right_bc[0, 0] = -1 / self.dx
        A_right_bc[0, 1] = 1 / self.dx

        A = np.zeros((4 * nx, 4 * nx))
        matrices = [A_left_bc, A_left_bc, A_right_bc, A_right_bc]
        for i in range(4):
            A[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = matrices[i]

        return A
    
    def construct_A_mixed(self):
        nx = self.nx
        def tridiag(a, b, c, k1=-1, k2=0, k3=1):
            return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

        small_diag = np.ones(nx - 1)
        main_diag = np.ones(nx)

        A_left_bc = tridiag(-1 * small_diag, 1 * main_diag, 0 * small_diag, -1, 0, 1) / self.dx
        # A_right_bc = -tridiag(-1 * small_diag, 1 * main_diag, 0 * small_diag, -1, 0, 1) / self.dx

        # A_left_bc = tridiag(0 * small_diag, -1 * main_diag, 1 * small_diag, -1, 0, 1) / self.dx
        A_right_bc = tridiag(-1/2 * small_diag, 0 * main_diag, 1/2 * small_diag, -1, 0, 1) / self.dx


        A_left_bc[0, 0] = 1 
        A_left_bc[0, 1] = 0
        A_left_bc[-1, -1] = -3/2 / self.dx
        A_left_bc[-1, -2] = 2 / self.dx
        A_left_bc[-1, -3] = -1/2 / self.dx

        A_right_bc[-1, -1] = 1
        A_right_bc[-1, -2] = 0
        A_right_bc[0, 0] = -3/2 / self.dx
        A_right_bc[0, 1] = 2 / self.dx
        A_right_bc[0, 2] = -1/2 / self.dx

        A = np.zeros((4 * nx, 4 * nx))
        matrices = [A_left_bc, A_left_bc, A_right_bc, A_right_bc]
        for i in range(4):
            A[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = matrices[i]

        return A
    
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
    
    
class FO_own():

    def __init__(self, deriv, x, y_right, y_right_dx, upwind=False):
        self.deriv = deriv
        # self.nx = 500

        self.x = x
        self.nx = len(self.x)
        self.y_right = y_right
        self.y_right_dx = y_right_dx

        if upwind:
            self.construct_A = self.construct_A_upwind
        else:
            self.construct_A = self.construct_A_central_diff
    
    def solve(self):
        self.dx = self.x[1] - self.x[0]
        initial_guess = 0.1 * np.ones((6, len(self.x)))
        
        dy = 1e50 * np.ones(initial_guess.shape)
        y = initial_guess

        while np.linalg.norm(dy) > 1e-12:
            print(np.linalg.norm(dy))

            A = self.construct_A()
            F = self.construct_F(self.x, y)
            J = self.construct_J(self.x, y)
            
            dy = np.linalg.solve(A - J, F - A @ y.flatten())
            dy = dy.reshape(6, self.nx)
            y = y + dy

        print(np.linalg.norm(dy))

        return self.x, y
    
    def construct_A_central_diff(self):
        nx = len(self.x)
        def tridiag(a, b, c, k1=-1, k2=0, k3=1):
            return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

        small_diag = np.ones(nx - 1)
        main_diag = np.ones(nx)

        A_left_bc = tridiag(-1/2 * small_diag, 0 * main_diag, 1/2 * small_diag, -1, 0, 1) / self.dx
        A_right_bc = tridiag(-1/2 * small_diag, 0 * main_diag, 1/2 * small_diag, -1, 0, 1) / self.dx

        A_left_bc[0, 0] = 1
        A_left_bc[0, 1] = 0
        A_left_bc[-1, -1] = -3/2 / self.dx
        A_left_bc[-1, -2] = 2 / self.dx
        A_left_bc[-1, -3] = -1/2 / self.dx

        A_right_bc[-1, -1] = 1 # h_c
        A_right_bc[-1, -2] = 0
        A_right_bc[0, 0] = -3/2 / self.dx
        A_right_bc[0, 1] = 2 / self.dx
        A_right_bc[0, 2] = -1/2 / self.dx



        A = np.zeros((6 * nx, 6 * nx))
        matrices = [A_left_bc, A_left_bc, A_left_bc, A_right_bc, A_right_bc, A_right_bc]

        
        for i in range(6):
            A[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = matrices[i]

        A[5*nx-1, 3*nx-1] = 2
        A[6*nx-1, 2*nx-1] = -2

        return A
    
    def construct_A_upwind(self):

        nx = len(self.x)
        def tridiag(a, b, c, k1=-1, k2=0, k3=1):
            return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

        small_diag = np.ones(nx - 1)
        main_diag = np.ones(nx)

        A_left_bc = tridiag(-1 * small_diag, 1 * main_diag, 0 * small_diag, -1, 0, 1) / self.dx
        # A_right_bc = -tridiag(-1 * small_diag, 1 * main_diag, 0 * small_diag, -1, 0, 1) / self.dx

        # A_left_bc = tridiag(0 * small_diag, -1 * main_diag, 1 * small_diag, -1, 0, 1) / self.dx
        A_right_bc = tridiag(0 * small_diag, -1 * main_diag, 1 * small_diag, -1, 0, 1) / self.dx

        A_left_bc[0, 0] = 1
        A_left_bc[0, 1] = 0
        if 0:
            A_left_bc[-1, -1] = 1 / self.dx
            A_left_bc[-1, -2] = -1 / self.dx
        if 0:
            A_left_bc[-1, -1] = -3/2 / self.dx
            A_left_bc[-1, -2] = 2 / self.dx
            A_left_bc[-1, -3] = -1/2 / self.dx
        if 1:
            A_left_bc[-1, -1] = -11/6 / self.dx
            A_left_bc[-1, -2] =  18/6 / self.dx
            A_left_bc[-1, -3] = - 9/6 / self.dx
            A_left_bc[-1, -4] =   2/6 / self.dx



        A_right_bc[-1, -1] = 1 # h_c
        A_right_bc[-1, -2] = 0
        if 1:
            A_right_bc[0, 0] = -1 / self.dx
            A_right_bc[0, 1] = 1 / self.dx
        if 0:
            A_right_bc[0, 0] = -3/2 / self.dx
            A_right_bc[0, 1] = 2 / self.dx
            A_right_bc[0, 2] = -1/2 / self.dx
        if 1:
            A_right_bc[0, 0] = -11/6 / self.dx
            A_right_bc[0, 1] =  18/6 / self.dx
            A_right_bc[0, 2] = - 9/6 / self.dx
            A_right_bc[0, 3] =   2/6 / self.dx



        A = np.zeros((6 * nx, 6 * nx))
        matrices = [A_left_bc, A_left_bc, A_left_bc, A_right_bc, A_right_bc, A_right_bc]

        
        for i in range(6):
            A[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = matrices[i]

        A[5*nx-1, 3*nx-1] = 2
        A[6*nx-1, 2*nx-1] = -2

        # print(sum(A[5*nx-1, :]))

        return A
    
    def construct_F(self, x , y):

        F = np.array(self.deriv(x, y))

        dz0_c   , dz0_s   , u0_c   , u0_s    = self.y_right
        dz0_c_dx, dz0_s_dx, u0_c_dx, u0_s_dx = self.y_right_dx


        # left boundary
        F[0, 0] = 0
        F[1, 0] = 0
        F[2, 0] = 0



        # right boundary
        F[3, -1] = 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s)
        F[4, -1] = 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s)
        F[5, -1] = 1 / 2 * (dz0_c * u0_s_dx + dz0_c_dx * u0_s + dz0_s * u0_c_dx + dz0_s_dx * u0_c)


        # print("here")
        # print(F[3, -1]) 
        # print(F[4, -1]) 
        # print(F[5, -1])

        


        F = F.flatten()

        return F
    
    def construct_J(self, x, y):
        y_flat = y.flatten()
        F = self.construct_F(x, y)
        J = np.zeros((len(y_flat), len(y_flat)))
        eps = 1e-7
        for i in range(len(y_flat)):
            y_flat[i] += eps
            F_new = self.construct_F(x, y_flat.reshape(6, self.nx))
            J[:, i] = (F_new - F) / eps
            y_flat[i] -= eps

        return J
   


counter = 0
class CSWEHarm_own():

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

        print(self.kappa)

        self.a_r = 1e-2
        self.r = 0.24

        self.epsilon = self.A / self.H
        self.eta = self.sigma * self.L / sqrt(self.g * self.H)
        self.U = self.epsilon * self.sigma * self.L

        self.small_number = 1e-5
        self.threshold = 0.995

    def h(self, x):
        return x
        # return 0.9*x

    def h_x(self, x):
        return 1
        # return 0.9
    
    def h_xx(self, x):
        return 0
    
    def solve_LO(self, upwind=False):

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
        
        my_LO = LO_own(deriv, upwind)
        x, y0 = my_LO.solve()
         
        self.dz0_c, self.dz0_s, self.u0_c, self.u0_s = y0

        self.x, self.y0 = x, y0

        # print(x.shape, y.shape)
        # tck = scipy.interpolate.splrep(x, y.T, s=0, k=3) # s=0 for no smoothing, k=3 for cubic
        # pp = scipy.interpolate.PPoly.from_spline(tck)
        # self.pol0 = pp

    def solve_FO(self, upwind=False):

        def deriv(x, vector):
            dz1_r, dz1_c, dz1_s, u1_r, u1_c, u1_s = vector


            dz0_c    , dz0_s    , u0_c    , u0_s = self.y0
            dz0_c_dx , dz0_s_dx , u0_c_dx , u0_s_dx = np.gradient(self.y0, self.x, axis=1, edge_order=2)

            # dz0_c_dxx, dz0_s_dxx, u0_c_dxx, u0_s_dxx = self.pol0(x, nu=2)
            
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


            du1_r = 1 / (1 - h) * (h_x * u1_r             - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s))
            du1_c = 1 / (1 - h) * (h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s))
            du1_s = 1 / (1 - h) * (h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s))
            
            if self.debug:
                a1 = h_x * u1_r - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s) # ( dz0_s * u0_s +  dz0_c * u0_c)
                a2 = h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s) # (-dz0_s * u0_s +  dz0_c * u0_c)
                a3 = h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s* u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s) # ( dz0_s * u0_c +  dz0_c * u0_s)
                print(a1[-3:], a2[-3:], a3[-3:])


            return [ddz1_r, ddz1_c, ddz1_s, du1_r, du1_c, du1_s]
    
        dz0_c    , dz0_s    , u0_c    , u0_s = self.y0
        dz0_c_dx , dz0_s_dx , u0_c_dx , u0_s_dx = np.gradient(self.y0, self.x, axis=1, edge_order=2)

        y_right = self.y0[:, -1]
        y_right_dx = np.sum(np.array([-1/2, 2, -3/2]) * self.y0[:, -1:-4:-1], axis=1) /  (self.x[1] - self.x[0])

        my_FO = FO_own(deriv, self.x, y_right, y_right_dx, upwind)
        x, y1 = my_FO.solve()

        self.x, self.y1 = x, y1
        





import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm


from coordinate_transform import heatmap


class DefinaSolution():
    def __init__(self, t, x, D_xt, u_xt):
        self.t = t
        self.x = x
        self.D_xt = D_xt
        self.u_xt = u_xt


    def heatmaps(self):
        start, end, step = 50000, 100000, 10

        T_mesh = np.tile(self.t[start:end:step], (len(self.x), 1))
        # X_mesh = np.tile((x_x*l_t).reshape(-1, 1), (1, len(t)))
        X_mesh = self.x[:, np.newaxis] * np.ones(self.t.shape)[start:end:step][np.newaxis, :]

        heatmap(T_mesh, X_mesh, self.D_xt[:, start:end:step] + X_mesh, "waterlevel(x,t)")
        heatmap(T_mesh, X_mesh, self.u_xt[:,  start:end:step], "u(x,t)")
        heatmap(T_mesh, X_mesh, self.D_xt[:, start:end:step] + X_mesh - np.mean(self.D_xt[:, start:end:step] + X_mesh, axis=0), \
                "deviation from mean (over x) waterlevel(x, t)")
        heatmap(T_mesh, X_mesh, abs(self.u_xt[:, start:end:step]), "abs(u(x,t))")
        



class Defina():

    def __init__(self):
        
        self.A = 0.84
        self.kappa = 1.7e1
        self.H = 12
        self.L = 1.9e4
        self.h0 = 0.0025
        # h0 = 0.00025
        self.omega = 1.4e-4

        self.nx = 500
        self.nr_periods = 6

        self.r = 0.24
        self.c_d = 0.01

        self.a_r = 3e-1

        self._update_u = self._update_u_linear_friction

    def generate_solution(self):
        return DefinaSolution(self.t, self.x, self.D_xt, self.u_xt)

    
    def solve_pde(self):

        # numerical precision
        self.dx = 1 / self.nx    
        self.dt = 0.9 / np.sqrt(self.kappa) * self.dx # CFL condition
        self.nt = int(self.nr_periods * 2 * pi / self.dt)

        # store the results here
        self.t = np.linspace(0, self.dt * self.nt, self.nt + 1)
        # self.x = np.linspace(-self.dx, 1.2 + self.dx, self.nx + 2) 
        self.x = np.arange(-self.dx, 1.2 + self.dx, self.dx) # in reality, 1.08 should be enough but 1.2 to be safe

        D_xt = np.zeros((len(self.x), len(self.t)))
        u_xt = np.zeros((len(self.x), len(self.t)))
        
        # initial conditions
        D_xt[:, 0] = (self.H + self.A) / self.H - self.x # since x = h_x

        for timestep in range(self.nt):
            # if timestep % 1000 == 0:
            #     print(timestep)
            D_xt[:, timestep + 1] = self._update_D(u_xt[:, timestep], D_xt[:, timestep], timestep * self.dt)
            u_xt[:, timestep + 1] = self._update_u(u_xt[:, timestep], self.x, D_xt[:, timestep], D_xt[:, timestep + 1])

        self.D_xt = D_xt
        self.u_xt = u_xt


    def _update_D(self, u_x, D_x, current_t):
        # A, H = c["A"], c["H"]

        eta_x = (1 + scipy.special.erf(2 * D_x / self.a_r))/ 2
        Y_x = eta_x * D_x + self.a_r / (4 * sqrt(pi)) * exp(-4 * (D_x / self.a_r)**2 )


        D_x2 = np.zeros(D_x.shape)
        D_x2[0] = self.A / self.H * np.cos(current_t) + 1
        D_x2[-1] = 0 # redundant ? since initialisation already sets it at zero ofc

        # print(u_x[-1], u_x[-2], u_x[-3])

        dYudx = -(-3 * Y_x[-1] * u_x[-1] + 4 * Y_x[-2] * u_x[-2] - Y_x[-3] * u_x[-3]) / (2*self.dx)
        D_x2[-1] = D_x[-1] - self.dt * dYudx / eta_x[-1]

        dYudx = (Y_x[2:] * u_x[2:] - Y_x[:-2] * u_x[:-2]) / (2 * self.dx)
        D_x2[1:-1] = (D_x[2:] + D_x[:-2]) / 2 - self.dt * dYudx / eta_x[1:-1]
        
        return D_x2
    

    # explicit, linear friction term
    def _update_u_linear_friction(self, u_x, h_x, D_x1, D_x2):
        # r, h0, kappa = c['r'], c['h0'], c['kappa']

        eta_x2 = (1 + scipy.special.erf(2 * D_x2 / self.a_r))/ 2
        Y_x2 = eta_x2 * D_x2 + self.a_r / (4 * sqrt(pi)) * exp(-4 * (D_x2 / self.a_r)**2 )

        Lambda = self.r / Y_x2


        u_x_t2 = np.zeros(u_x.shape)

        dudx = (-3 * u_x[0] + 4 * u_x[1] - u_x[2]) / (2*self.dx)
        dDdx = (-3 * D_x1[0] + 4 * D_x1[1] - D_x1[2]) / (2*self.dx)
        dhdx = (-3 * h_x[0] + 4 * h_x[1] - h_x[2]) / (2*self.dx)

        u_x_t2[0]    = (
                            u_x[0] \
                            - self.dt * u_x[0] * dudx 
                            - self.dt * self.kappa * (dDdx + dhdx)
                        ) / (1 + Lambda[0] * self.dt)

        dudx = -(-3 *  u_x[-1] + 4 *  u_x[-2] -  u_x[-3]) / (2*self.dx)
        dDdx = -(-3 * D_x1[-1] + 4 * D_x1[-2] - D_x1[-3]) / (2*self.dx)
        dhdx = -(-3 *  h_x[-1] + 4 *  h_x[-2] -  h_x[-3]) / (2*self.dx)

        # u_x_t2[-1]   = (
        #                     u_x[-1] \
        #                     - self.dt * u_x[-1] * dudx # this is zero???
        #                     - self.dt * self.kappa * (dDdx + dhdx)
        #                 ) / (1 + Lambda[-1] * self.dt)

        u_x_t2[-1] = 0
        

        dudx = (u_x[2:] - u_x[:-2]) / (2*self.dx)
        dDdx = (D_x1[2:] - D_x1[:-2]) / (2*self.dx)
        dhdx = (h_x[2:] - h_x[:-2]) / (2*self.dx)

        u_x_t2[1:-1] = (
                            (u_x[2:] + u_x[:-2]) / 2 \
                            - self.dt * u_x[1:-1] * dudx \
                            - self.dt * self.kappa * (dDdx + dhdx)
                        ) / (1 + Lambda[1:-1] * self.dt)
        
        return u_x_t2




if __name__ == "__main__":


    # small test
    computation = Defina()
    computation.solve_pde()
    sol = computation.generate_solution()

    sol.heatmaps()



import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm

from plots import *


class CoordinateTransformSolution():

    def __init__(self, t, x, l_t, D_xt, u_xt):
        self.t = t
        self.x = x
        self.l_t = l_t
        self.D_xt = D_xt
        self.u_xt = u_xt


    def heatmaps(self):
        print(len(self.t))

        start, end, step = 50000, 100000, 10
        start, end, stop = 0, -1, 1

        T_mesh = np.tile(self.t[start:end:step], (len(self.x), 1))
        # X_mesh = np.tile((x_x*l_t).reshape(-1, 1), (1, len(t)))
        X_mesh = self.x[:, np.newaxis] * self.l_t[start:end:step][np.newaxis, :]

        heatmap(T_mesh, X_mesh, self.D_xt[:, start:end:step] + X_mesh, "waterlevel(x,t)")
        heatmap(T_mesh, X_mesh, self.u_xt[:,  start:end:step], "u(x,t)")
        heatmap(T_mesh, X_mesh, self.D_xt[:, start:end:step] + X_mesh - np.mean(self.D_xt[:, start:end:step] + X_mesh, axis=0), \
                "deviation from mean (over x) waterlevel(x, t)")
        heatmap(T_mesh, X_mesh, abs(self.u_xt[:, start:end:step]), "abs(u(x,t))")


    def other_plots(self):
        plt.plot(self.t, self.l_t); plt.xlabel("x"); plt.ylabel("l(t)"); plt.show()

        spatial_aggregates(self.x, self.u_xt, "velocity")
        spatial_aggregates(self.x, self.D_xt + self.x[:, np.newaxis] * self.l_t[np.newaxis, :], "waterlevel")

        t_snapshots(self.x, self.t, self.u_xt, "velocity")
        t_snapshots(self.x, self.t, self.D_xt + self.x[:, np.newaxis] * self.l_t[np.newaxis, :], "waterlevel")

        # x_snapshots(self.x, self.t, self.u_xt, "velocity")
        # x_snapshots(self.x, self.t, self.D_xt + self.x[:, np.newaxis] * self.l_t[np.newaxis, :], "waterlevel")



    def one_period_solution(self, period_n = 2):
        start_t = period_n * (2*pi)
        end_t = start_t + 2 * pi

        t1_idx = np.argmin(np.abs(self.t - start_t))
        t2_idx = np.argmin(np.abs(self.t - end_t))

        pt = self.t[t1_idx:t2_idx]
        px = self.x
        pl_t = self.l_t[t1_idx:t2_idx]
        pD_xt = self.D_xt[:, t1_idx:t2_idx]
        pu_xt = self.u_xt[:, t1_idx:t2_idx] 

        p_sol = CoordinateTransformSolution(pt, px, pl_t, pD_xt, pu_xt)
        return p_sol
    

class CoordinateTransform():

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

        self._update_u = self._update_u_linear_friction

    def generate_solution(self):
        return CoordinateTransformSolution(self.t, self.x, self.l_t, self.D_xt, self.u_xt)

    
    def solve_pde(self):

        # numerical precision
        self.dx = 1 / self.nx    
        self.dt = 0.9 / np.sqrt(self.kappa) * self.dx # CFL condition
        self.nt = int(self.nr_periods * 2 * pi / self.dt)

        # store the results here
        self.t = np.linspace(0, self.dt * self.nt, self.nt + 1)
        self.x = np.linspace(-self.dx, 1 + self.dx, self.nx + 2)
        
        self.l_t = np.zeros(self.nt + 1)
        self.D_xt = np.zeros((self.nx + 2, self.nt + 1))
        self.u_xt = np.zeros((self.nx + 2, self.nt + 1))
        
        # initial conditions
        self.D_xt[:, 0] = (self.H + self.A) / self.H - self.x # since x = h_x
        self.l_t[0] = 1

        for t_i in range(self.nt):
            # if timestep % 1000 == 0:
            #     print(timestep)
            self._update_l(t_i)
            self._update_D(t_i)
            self._update_u(t_i)

            # l_t[timestep + 1] = self._update_l(l_t[timestep], u_xt[:, timestep])
            # D_xt[:, timestep + 1] = self._update_D(l_t[timestep], l_t[timestep + 1], u_xt[:, timestep], D_xt[:, timestep], timestep * self.dt)
            # u_xt[:, timestep + 1] = self._update_u(l_t[timestep], l_t[timestep + 1], u_xt[:, timestep], self.x * l_t[timestep], D_xt[:, timestep], D_xt[:, timestep + 1])

    def _update_l(self, t_i):
        self.l_t[t_i + 1] = self.l_t[t_i] + self.u_xt[-2, t_i] * self.dt


    def _update_D(self, t_i):

        current_t = self.t[t_i]

        D_x2 = self.D_xt[:, t_i + 1] # immediately set them here
        l_2 = self.l_t[t_i + 1]
        
        D_x = self.D_xt[:, t_i]
        u_x = self.u_xt[:, t_i]
        l_1 = self.l_t[t_i]
        
        D_x2[0] = self.A / self.H * np.cos(current_t) + 1
        D_x2[-1] = 0 # redundant ?

        dDdx =  (D_x[2:] - D_x[:-2]) / (2 * self.dx)
        dDudx = (D_x[2:] * u_x[2:] - D_x[:-2] * u_x[:-2]) / (2 * self.dx)

        D_x2[1:-1] = (D_x[2:] + D_x[:-2]) / 2 \
                        + self.x[1:-1] / l_1 * (l_2 - l_1) * dDdx \
                        - self.dt / l_1 * dDudx
    

    # explicit, linear friction term
    def _update_u_linear_friction(self, t_i):

        D_x2 = self.D_xt[:, t_i + 1] # immediately set them here
        u_x2 = self.u_xt[:, t_i + 1]
        l_2 = self.l_t[t_i + 1]
        
        D_x = self.D_xt[:, t_i]
        u_x = self.u_xt[:, t_i]
        l_1 = self.l_t[t_i]

        h_x = self.x * self.l_t[t_i]
        
        Lambda = self.r / (D_x2 + self.h0)

        dudx = (-3 * u_x[0] + 4 * u_x[1] - u_x[2]) / (2*self.dx)
        dDdx = (-3 * D_x[0] + 4 * D_x[1] - D_x[2]) / (2*self.dx)
        dhdx = (-3 * h_x[0] + 4 * h_x[1] - h_x[2]) / (2*self.dx)

        u_x2[0]    = (
                            u_x[0] \
                            - self.dt / l_1 * u_x[0] * dudx 
                            - self.dt / l_1 * self.kappa * (dDdx + dhdx)
                        ) / (1 + Lambda[0] * self.dt)

        dudx = -(-3 * u_x[-1] + 4 * u_x[-2] - u_x[-3]) / (2*self.dx)
        dDdx = -(-3 * D_x[-1] + 4 * D_x[-2] - D_x[-3]) / (2*self.dx)
        dhdx = -(-3 * h_x[-1] + 4 * h_x[-2] - h_x[-3]) / (2*self.dx)

        u_x2[-1]   = (
                            u_x[-1] \
                            + self.dt / l_1 * ((l_2 - l_1) / self.dt - u_x[-1]) * dudx # this is zero???
                            - self.dt / l_1 * self.kappa * (dDdx + dhdx)
                        ) / (1 + Lambda[-1] * self.dt)
        

        dudx = (u_x[2:] - u_x[:-2]) / (2*self.dx)
        dDdx = (D_x[2:] - D_x[:-2]) / (2*self.dx)
        dhdx = (h_x[2:] - h_x[:-2]) / (2*self.dx)

        u_x2[1:-1] = (
                            (u_x[2:] + u_x[:-2]) / 2 \
                            + self.dt / l_1 * (self.x[1:-1] * (l_2 - l_1) / self.dt - u_x[1:-1]) * dudx \
                            - self.dt / l_1 * self.kappa * (dDdx + dhdx)
                        ) / (1 + Lambda[1:-1] * self.dt)
        

    def _update_u_quadratic_friction(self, t_i):

        D_x2 = self.D_xt[:, t_i + 1] # immediately set them here
        l_2 = self.l_t[t_i + 1]
        
        D_x = self.D_xt[:, t_i]
        u_x = self.u_xt[:, t_i]
        l_1 = self.l_t[t_i]

        h_x = self.x * self.l_t[t_i]

        u_x_old_guess = 1e100*np.ones(u_x.shape) # to make sure there is a large error
        u_x_guess = u_x

        while np.linalg.norm(u_x_guess - u_x_old_guess) > 1e-8:
            # print(np.linalg.norm(u_x_t2 - u_x_guess)) # usually converges in 2/3/4 steps 

            u_x_old_guess = u_x_guess
            Lambda = self.c_d * np.abs(u_x_guess) / (D_x2 + self.h0)**(4/3)


            u_x_guess = np.zeros(u_x.shape)

            dudx = (-3 * u_x[0] + 4 * u_x[1] - u_x[2]) / (2*self.dx)
            dDdx = (-3 * D_x[0] + 4 * D_x[1] - D_x[2]) / (2*self.dx)
            dhdx = (-3 * h_x[0] + 4 * h_x[1] - h_x[2]) / (2*self.dx)

            u_x_guess[0]    = (
                                u_x[0] \
                                + self.dt / l_1 * u_x[0] * dudx 
                                - self.dt / l_1 * self.kappa * (dDdx + dhdx)
                            ) / (1 + Lambda[0] * self.dt)

            dudx = -(-3 * u_x[-1] + 4 * u_x[-2] - u_x[-3]) / (2*self.dx)
            dDdx = -(-3 * D_x[-1] + 4 * D_x[-2] - D_x[-3]) / (2*self.dx)
            dhdx = -(-3 * h_x[-1] + 4 * h_x[-2] - h_x[-3]) / (2*self.dx)

            u_x_guess[-1]   = (
                                u_x[-1] \
                                + self.dt / l_1 * ((l_2 - l_1) / self.dt - u_x[-1]) * dudx # this is zero???
                                - self.dt / l_1 * self.kappa * (dDdx + dhdx)
                            ) / (1 + Lambda[-1] * self.dt)
            

            dudx = (u_x[2:] - u_x[:-2]) / (2*self.dx)
            dDdx = (D_x[2:] - D_x[:-2]) / (2*self.dx)
            dhdx = (h_x[2:] - h_x[:-2]) / (2*self.dx)

            u_x_guess[1:-1] = (
                                (u_x[2:] + u_x[:-2]) / 2 \
                                + self.dt / l_1 * (x[1:-1] * (l_2 - l_1) / self.dt - u_x[1:-1]) * dudx \
                                - self.dt / l_1 * self.kappa * (dDdx + dhdx)
                            ) / (1 + Lambda[1:-1] * self.dt)
        
        self.u_xt[:, t_i + 1] = u_x_guess



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib import cm

    # small test
    computation = CoordinateTransform()
    computation.solve_pde()

    t, x, l_t, D_xt, u_xt = computation.t, computation.x, computation.l_t, computation.D_xt, computation.u_xt
    plt.plot(t / computation.omega / 3600, l_t * computation.L / 1000)


    computation.r = 0.05
    computation.solve_pde()

    t, x, l_t, D_xt, u_xt = computation.t, computation.x, computation.l_t, computation.D_xt, computation.u_xt
    plt.plot(t / computation.omega / 3600, l_t * computation.L / 1000)

    computation._update_u = computation._update_u_quadratic_friction
    computation.solve_pde()

    t, x, l_t, D_xt, u_xt = computation.t, computation.x, computation.l_t, computation.D_xt, computation.u_xt
    plt.plot(t / computation.omega / 3600, l_t * computation.L / 1000)

    plt.xlabel("time t [hours]")
    plt.ylabel("estuary length l(t) [km]")

    plt.show()
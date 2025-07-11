

import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm


from plots import *


class DefinaSolution():
    def __init__(self, t, x, D_xt, u_xt, eta_xt, Y_xt):
        self.t = t
        self.x = x
        self.D_xt = D_xt
        self.u_xt = u_xt
        self.eta_xt = eta_xt
        self.Y_xt = Y_xt


    def heatmaps(self):
        start, end, step = 50000, 100000, 10
        # start, end, step = 0, 1000, 1
        start, end, step = 0, 27070, 10


        T_mesh = np.tile(self.t[start:end:step], (len(self.x), 1))
        # X_mesh = np.tile((x_x*l_t).reshape(-1, 1), (1, len(t)))
        X_mesh = self.x[:, np.newaxis] * np.ones(self.t.shape)[start:end:step][np.newaxis, :]

        heatmap(T_mesh, X_mesh, self.D_xt[:, start:end:step] + X_mesh, "waterlevel(x,t)")
        heatmap(T_mesh, X_mesh, self.u_xt[:,  start:end:step], "u(x,t)")
        heatmap(T_mesh, X_mesh, self.D_xt[:, start:end:step] + X_mesh - np.mean(self.D_xt[:, start:end:step] + X_mesh, axis=0), \
                "deviation from mean (over x) waterlevel(x, t)")
        heatmap(T_mesh, X_mesh, abs(self.u_xt[:, start:end:step]), "abs(u(x,t))")

        heatmap(T_mesh, X_mesh, self.Y_xt[:, start:end:step], "Y(x,t)")
        heatmap(T_mesh, X_mesh, self.eta_xt[:, start:end:step], "$\\eta(x,t)$")

    
    def other_plots(self):

        spatial_aggregates(self.x, self.u_xt, "velocity")
        spatial_aggregates(self.x, self.D_xt + self.x[:, np.newaxis] * np.ones(self.t.shape)[np.newaxis, :], "waterlevel")

        t_snapshots(self.x, self.t, self.u_xt, "velocity")
        t_snapshots(self.x, self.t, self.D_xt + self.x[:, np.newaxis] * np.ones(self.t.shape)[np.newaxis, :], "waterlevel")

        # x_snapshots(self.x, self.t, self.u_xt, "velocity")
        # x_snapshots(self.x, self.t, self.D_xt + self.x[:, np.newaxis] * np.ones(self.t.shape)[np.newaxis, :], "waterlevel")



    def one_period_solution(self, period_n = 2):
        start_t = period_n * (2*pi)
        end_t = start_t + 2 * pi

        t1_idx = np.argmin(np.abs(self.t - start_t))
        t2_idx = np.argmin(np.abs(self.t - end_t))

        pt = self.t[t1_idx:t2_idx]
        px = self.x
        pD_xt = self.D_xt[:, t1_idx:t2_idx]
        pu_xt = self.u_xt[:, t1_idx:t2_idx] 
        peta_xt = self.eta_xt[:, t1_idx:t2_idx]
        pY_xt = self.Y_xt[:, t1_idx:t2_idx]

        p_sol = DefinaSolution(pt, px, pD_xt, pu_xt, peta_xt, pY_xt)
        return p_sol
    


            
        



class Defina():

    def __init__(self):
        
        self.A = 0.84
        self.kappa = 1.7e1
        self.H = 12
        self.L = 1.9e4

        self.dL = 0.5

        self.h0 = 0.0025
        # h0 = 0.00025
        self.omega = 1.4e-4

        self.nx = 500
        self.nr_periods = 6

        self.r = 0.5
        self.c_d = 0.03

        self.a_r = 5e-3

        self._update_u = self._update_u_linear_friction


    def generate_solution(self):
        return DefinaSolution(self.t, self.x, self.D_xt, self.u_xt, self.eta_xt, self.Y_xt)
    

    
    def solve_pde(self):

        # numerical precision
        self.dx = 1 / self.nx    
        self.dt = 0.9 / np.sqrt(self.kappa) * self.dx # / 10 # CFL condition 
        self.nt = int(self.nr_periods * 2 * pi / self.dt)

        # store the results here
        self.t = np.linspace(0, self.dt * self.nt, self.nt + 1)
        self.x = np.arange(-self.dx, 1 + self.dL + self.dx, self.dx) # in reality, 1.08 should be enough but 1.2 or 1.5 to be safe

        self.h_x = self.x.copy() # constant bed


        self.D_xt = np.zeros((len(self.x), len(self.t)))
        self.u_xt = np.zeros((len(self.x), len(self.t)))
        self.eta_xt = np.zeros((len(self.x), len(self.t)))
        self.Y_xt = np.zeros((len(self.x), len(self.t)))

        self.clipped = np.zeros(self.eta_xt.shape)
        
        # initial conditions
        self.D_xt[:, 0] = (self.H + self.A) / self.H - self.x # since x = h_x
        
        self.eta_xt[:, 0] = (1 + scipy.special.erf(2 * self.D_xt[:, 0] / self.a_r))/ 2
        # threshold = 1e-20
        threshold = 1e-8

        self.eta_xt[:, 0] = np.clip(self.eta_xt[:, 0], threshold, None)

        self.Y_xt[:, 0] = self.eta_xt[:, 0] * self.D_xt[:, 0] + self.a_r / (4 * sqrt(pi)) * exp(-4 * (self.D_xt[:, 0] / self.a_r)**2 )



        for t_i in range(self.nt):

            # if timestep < 100:
            #     print(self.D_xt[-100:, timestep])
            #     print(self.eta_xt[-100:, timestep])
            #     print(self.Y_xt[-100:, timestep])
            # if timestep % 1000 == 0:
            #     print(timestep)

            self._update_D(t_i)
            self._update_eta(t_i)
            self.clipped[:, t_i + 1] = self.eta_xt[:, t_i + 1] < threshold
            self.eta_xt[:, t_i + 1] = np.clip(self.eta_xt[:, t_i + 1], threshold, None)
            self._update_Y(t_i)
            self._update_u(t_i)



    def _update_eta(self, t_i):
        D_x2 = self.D_xt[:, t_i + 1]
        eta_x2 = (1 + scipy.special.erf(2 * D_x2 / self.a_r)) / 2

        self.eta_xt[:, t_i + 1] = eta_x2
    

    def _update_Y(self, t_i):
        D_x2 = self.D_xt[:, t_i + 1]
        eta_x2 = self.eta_xt[:, t_i + 1]

        s1 = (D_x2 / self.a_r)**2
        if (s1 > 1e9).any(): 
            print("here")
            print(t_i)
            p_i = np.argmax(abs(s1))
            print(np.argmax(abs(s1)))

            print(self.eta_xt[p_i - 5:p_i + 5, t_i - 1])
            print(self.eta_xt[p_i - 5:p_i + 5, t_i])
            print(self.eta_xt[p_i - 5:p_i + 5, t_i + 1])

            print("-" * 5)
            print(self.D_xt[p_i - 5:p_i + 5, t_i - 2])
            print(self.D_xt[p_i - 5:p_i + 5, t_i - 1])
            print(self.D_xt[p_i - 5:p_i + 5, t_i])
            print(self.D_xt[p_i - 5:p_i + 5, t_i + 1])

            print("-" * 5)
            print(self.clipped[p_i - 5:p_i + 5, t_i - 2])
            print(self.clipped[p_i - 5:p_i + 5, t_i - 1])
            print(self.clipped[p_i - 5:p_i + 5, t_i])
            print(self.clipped[p_i - 5:p_i + 5, t_i + 1])


            t_condition = self.t < 15
            x_condition = np.logical_and(self.x < 1.15, self.x > 0.95)


            tmesh, xmesh = np.meshgrid(self.t[t_condition], self.x[x_condition])
            fig, ax = plt.subplots(figsize=(20, 20))

            im = ax.pcolormesh(tmesh, xmesh, self.clipped[x_condition].T[t_condition].T, cmap='inferno', shading='nearest')
            plt.show()


            raise SystemError


            plt.plot(self.eta_xt[p_i - 5:p_i + 5, t_i-4:t_i+2], 'o', linewidth=0)
            plt.legend(range(7))
            plt.show()
            plt.semilogy(self.D_xt[p_i - 5:p_i + 5, t_i-4:t_i+2], 'o', linewidth=0)
            plt.legend(range(7))
            plt.show()
            
            print(len(s1))
            print(D_x2)
            print(s1)
            print(self.a_r)
            raise SystemError
        p1 =  exp(-4 *  s1)
        Y_x2 = eta_x2 * D_x2 + self.a_r / (4 * sqrt(pi)) * p1
        self.Y_xt[:, t_i + 1] = Y_x2


    def _update_D(self, t_i):

        current_t = self.t[t_i]

        D_x2 = self.D_xt[:, t_i + 1] # immediately set them here
        
        D_x = self.D_xt[:, t_i]
        eta_x = self.eta_xt[:, t_i]
        Y_x = self.Y_xt[:, t_i]
        u_x = self.u_xt[:, t_i]



        D_x2[0] = self.A / self.H * np.cos(current_t) + 1
        D_x2[-1] = 0 # redundant ? since initialisation already sets it at zero ofc

        dYudx = -(-3 * Y_x[-1] * u_x[-1] + 4 * Y_x[-2] * u_x[-2] - Y_x[-3] * u_x[-3]) / (2*self.dx)
        D_x2[-1] = D_x[-1] - self.dt * dYudx / eta_x[-1]

        dYudx = (Y_x[2:] * u_x[2:] - Y_x[:-2] * u_x[:-2]) / (2 * self.dx)
        D_x2[1:-1] = (D_x[2:] + D_x[:-2]) / 2 - self.dt * dYudx / eta_x[1:-1]
        
        # self.D_xt[:, t_i + 1] = D_x2
    

    def _update_u_linear_friction(self, t_i):
        current_t = self.t[t_i]

        D_x = self.D_xt[:, t_i]
        Y_x2 = self.Y_xt[:, t_i + 1]
        u_x = self.u_xt[:, t_i]
        u_x2 = self.u_xt[:, t_i + 1]


        Lambda = self.r / Y_x2

        dudx = (-3 * u_x[0] + 4 * u_x[1] - u_x[2]) / (2*self.dx)
        dDdx = (-3 * D_x[0] + 4 * D_x[1] - D_x[2]) / (2*self.dx)
        dhdx = (-3 * self.h_x[0] + 4 * self.h_x[1] - self.h_x[2]) / (2*self.dx)

        u_x2[0]    = (
                            u_x[0] \
                            - self.dt * u_x[0] * dudx 
                            - self.dt * self.kappa * (dDdx + dhdx)
                        ) / (1 + Lambda[0] * self.dt)

        dudx = -(-3 * u_x[-1] + 4 * u_x[-2] - u_x[-3]) / (2*self.dx)
        dDdx = -(-3 * D_x[-1] + 4 * D_x[-2] - D_x[-3]) / (2*self.dx)
        dhdx = -(-3 * self.h_x[-1] + 4 * self.h_x[-2] - self.h_x[-3]) / (2*self.dx)

        # u_x_t2[-1]   = (
        #                     u_x[-1] \
        #                     - self.dt * u_x[-1] * dudx # this is zero???
        #                     - self.dt * self.kappa * (dDdx + dhdx)
        #                 ) / (1 + Lambda[-1] * self.dt)

        u_x2[-1] = 0
        

        dudx = (u_x[2:] - u_x[:-2]) / (2*self.dx)
        dDdx = (D_x[2:] - D_x[:-2]) / (2*self.dx)
        dhdx = (self.h_x[2:] - self.h_x[:-2]) / (2*self.dx)

        u_x2[1:-1] = (
                            (u_x[2:] + u_x[:-2]) / 2 \
                            - self.dt * u_x[1:-1] * dudx \
                            - self.dt * self.kappa * (dDdx + dhdx)
                        ) / (1 + Lambda[1:-1] * self.dt)
        
    


if __name__ == "__main__":


    # small test
    computation = Defina()
    computation.solve_pde()
    sol = computation.generate_solution()

    sol.heatmaps()

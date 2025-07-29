
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
    

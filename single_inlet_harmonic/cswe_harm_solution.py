
import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm


# from single_inlet_moving_boundary.plots import *
from plots import *

class CSWEHarmSolution():

    def __init__(self, t, x, dz0_xt, u0_xt, dz1_xt, u1_xt, dz_xt, u_xt):
        self.t = t
        self.x = x
        self.dz0_xt = dz0_xt
        self.u0_xt = u0_xt

        self.dz1_xt = dz1_xt
        self.u1_xt = u1_xt

        self.dz_xt = dz_xt
        self.u_xt = u_xt
    

    def heatmaps(self):
        start, end, step = 0, -1, 10

        T_mesh = np.tile(self.t[start:end:step], (len(self.x), 1))
        # X_mesh = np.tile((x_x*l_t).reshape(-1, 1), (1, len(t)))
        X_mesh = self.x[:, np.newaxis] * np.ones(len(self.t))[start:end:step][np.newaxis, :]

        heatmap(T_mesh, X_mesh, self.dz0_xt[:, start:end:step], "dzeta0(x,t)")
        heatmap(T_mesh, X_mesh, self.u0_xt[:,  start:end:step], "u0(x,t)")

        heatmap(T_mesh, X_mesh, self.dz1_xt[:, start:end:step], "dzeta1(x,t)")
        heatmap(T_mesh, X_mesh, self.u1_xt[:,  start:end:step], "u1(x,t)")

        heatmap(T_mesh, X_mesh, self.dz_xt[:, start:end:step], "dzeta(x,t)")
        heatmap(T_mesh, X_mesh, self.u_xt[:,  start:end:step], "u(x,t)")


    def other_plots(self):
        # plt.plot(self.t, self.l_t); plt.xlabel("x"); plt.ylabel("l(t)"); plt.show()

        spatial_aggregates(self.x, self.u_xt, "velocity")
        spatial_aggregates(self.x, self.D_xt + self.x[:, np.newaxis] * self.l_t[np.newaxis, :], "waterlevel")

        t_snapshots(self.x, self.t, self.u_xt, "velocity")
        t_snapshots(self.x, self.t, self.D_xt + self.x[:, np.newaxis] * self.l_t[np.newaxis, :], "waterlevel")

        # x_snapshots(self.x, self.t, self.u_xt, "velocity")
        # x_snapshots(self.x, self.t, self.D_xt + self.x[:, np.newaxis] * self.l_t[np.newaxis, :], "waterlevel")

    

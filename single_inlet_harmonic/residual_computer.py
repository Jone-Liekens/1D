

import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt, zeros, ones, nan
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm




class ResidualComputer():

    def __init__(self, obj):
        self.r = obj.r
        self.h0 = obj.h0
        self.r = obj.r
        self.kappa = obj.kappa

    def residual1(self, x, y0, y1):

        dz0_c    , dz0_s    , u0_c    , u0_s = y0
        dz0_c_dx , dz0_s_dx , u0_c_dx , u0_s_dx = np.gradient(y0, x, axis=1, edge_order=2)

        h, h_x = x, 1
        
        dz1_r,  dz1_c,  dz1_s,  u1_r,  u1_c,  u1_s  = y1
        ddz1_r, ddz1_c, ddz1_s, du1_r, du1_c, du1_s = np.gradient(y1, x, axis=1, edge_order=2)

        res = zeros(y1.shape)
        # momentum equations
        res[0] = ddz1_r - (1 / (1 - h + self.h0) * (
            - self.r * u1_r
            - 1 / 2 * (  dz0_c *  u0_s - dz0_s *  u0_c)
            - 1 / 2 * (  dz0_s * dz0_s_dx + dz0_c * dz0_c_dx) * self.kappa
        ) - 1/2 * (u0_c * u0_c_dx + u0_s * u0_s_dx)) / self.kappa
        res[1] = ddz1_c - (1 / (1 - h + self.h0) * (
            - self.r * u1_c
            - 1 / 2 * (  dz0_c *  u0_s + dz0_s *  u0_c)
            - 1 / 2 * (- dz0_s * dz0_s_dx + dz0_c * dz0_c_dx) * self.kappa
        ) - 1/2 * (u0_c * u0_c_dx - u0_s * u0_s_dx) - 2 * u1_s) / self.kappa
        res[2] = ddz1_s - (1 / (1 - h + self.h0) * (
            - self.r * u1_s
            + 1 / 2 * (  dz0_c *  u0_c - dz0_s *  u0_s)
            - 1 / 2 * (  dz0_c * dz0_s_dx + dz0_s * dz0_c_dx) * self.kappa
        ) - 1/2 * (u0_c * u0_s_dx + u0_s * u0_c_dx) + 2 * u1_c) / self.kappa


        res[3] = du1_r - 1 / (1 - h) * (h_x * u1_r             - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s))
        res[4] = du1_c - 1 / (1 - h) * (h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s))
        res[5] = du1_s - 1 / (1 - h) * (h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s))

        fig, axs = plt.subplots(1, 6, figsize=(30, 5))
        for i in range(6):
            axs[i].plot(x[1:-1], res[i, 1:-1])
        plt.show()

        for i in range(6):
            for k in range(20):
                val = res[i, -1-k]
                print(str(val), (25 - len(str(val))) * ' ', end='')
            print("")

    def residual2(self, x, y0, y1):

        y0_ders = [0, 0, 0, 0]
        for i in range(4):
            tck = scipy.interpolate.splrep(x, y0[i], s=0, k=3) # s=0 for no smoothing, k=3 for cubic
            pp = scipy.interpolate.PPoly.from_spline(tck)
            y0_ders[i] = pp(x, nu=1)


        y1_ders = [0, 0, 0, 0, 0, 0]
        for i in range(6):
            tck = scipy.interpolate.splrep(x, y1[i], s=0, k=3) # s=0 for no smoothing, k=3 for cubic
            pp = scipy.interpolate.PPoly.from_spline(tck)

            plt.plot(x, y1[i])
            plt.show()
            plt.plot(x, pp(x, nu=1))
            plt.show()

            y1_ders[i] = pp(x, nu=1)

    

        dz0_c    , dz0_s    , u0_c    , u0_s = y0
        dz0_c_dx , dz0_s_dx , u0_c_dx , u0_s_dx = y0_ders

        h, h_x = x, 1
        
        dz1_r,  dz1_c,  dz1_s,  u1_r,  u1_c,  u1_s  = y1
        ddz1_r, ddz1_c, ddz1_s, du1_r, du1_c, du1_s =  y1_ders

        res = zeros(y1.shape)
        # momentum equations
        res[0] = ddz1_r - (1 / (1 - h + self.h0) * (
            - self.r * u1_r
            - 1 / 2 * (  dz0_c *  u0_s - dz0_s *  u0_c)
            - 1 / 2 * (  dz0_s * dz0_s_dx + dz0_c * dz0_c_dx) * self.kappa
        ) - 1/2 * (u0_c * u0_c_dx + u0_s * u0_s_dx)) / self.kappa
        res[1] = ddz1_c - (1 / (1 - h + self.h0) * (
            - self.r * u1_c
            - 1 / 2 * (  dz0_c *  u0_s + dz0_s *  u0_c)
            - 1 / 2 * (- dz0_s * dz0_s_dx + dz0_c * dz0_c_dx) * self.kappa
        ) - 1/2 * (u0_c * u0_c_dx - u0_s * u0_s_dx) - 2 * u1_s) / self.kappa
        res[2] = ddz1_s - (1 / (1 - h + self.h0) * (
            - self.r * u1_s
            + 1 / 2 * (  dz0_c *  u0_c - dz0_s *  u0_s)
            - 1 / 2 * (  dz0_c * dz0_s_dx + dz0_s * dz0_c_dx) * self.kappa
        ) - 1/2 * (u0_c * u0_s_dx + u0_s * u0_c_dx) + 2 * u1_c) / self.kappa


        res[3] = du1_r - 1 / (1 - h) * (h_x * u1_r             - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c + dz0_s * u0_s_dx + dz0_s_dx * u0_s))
        res[4] = du1_c - 1 / (1 - h) * (h_x * u1_c - 2 * dz1_s - 1 / 2 * (dz0_c * u0_c_dx + dz0_c_dx * u0_c - dz0_s * u0_s_dx - dz0_s_dx * u0_s))
        res[5] = du1_s - 1 / (1 - h) * (h_x * u1_s + 2 * dz1_c - 1 / 2 * (dz0_s * u0_c_dx + dz0_s_dx * u0_c + dz0_c * u0_s_dx + dz0_c_dx * u0_s))

        fig, axs = plt.subplots(1, 6, figsize=(30, 5))
        for i in range(6):
            axs[i].plot(x[1:-1], res[i, 1:-1])
        plt.show()

        for i in range(6):
            for k in range(20):
                val = res[i, -1-k]
                print(str(val), (25 - len(str(val))) * ' ', end='')
            print("")
        
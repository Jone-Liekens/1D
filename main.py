
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean
import scipy

# constants:
A = 0.84
kappa = 1.7e1
H = 12
L = 1.9e4
h0 = 0.0025
# h0 = 0.00025
omega = 1.4e-4

r = 0.24


def update_l(l__t, u_x_t, dt):
    return l__t + u_x_t[-2] * dt

def update_D(l__t1, l__t2, u_x_t, x_x, D_x_t, dt, dx, t, A, H):
    D_x_t2 = np.zeros(D_x_t.shape)
    D_x_t2[0] = A / H * np.cos(t) + 1
    D_x_t2[-1] = 0 # redundant ?

    dDdx =  (D_x_t[2:] - D_x_t[:-2]) / (2 * dx)
    dDudx = (D_x_t[2:] * u_x_t[2:] - D_x_t[:-2] * u_x_t[:-2]) / (2 * dx)

    D_x_t2[1:-1] = (D_x_t[2:] + D_x_t[:-2]) / 2 \
                    + x_x[1:-1] / l__t1 * (l__t2 - l__t1) * dDdx \
                    - dt / l__t1 * dDudx
    
    return D_x_t2

# explicit, linear friction term
def update_u1(l__t1, l__t2, u_x_t, x_x, h_x, D_x_t, D_x_t2, dt, dx, r, h0, kappa):
    Lambda = r / (D_x_t2 + h0)

    u_x_t2 = np.zeros(u_x_t.shape)

    dudx = (-3 * u_x_t[0] + 4 * u_x_t[1] - u_x_t[2]) / (2*dx)
    dDdx = (-3 * D_x_t[0] + 4 * D_x_t[1] - D_x_t[2]) / (2*dx)
    dhdx = (-3 * h_x[0] + 4 * h_x[1] - h_x[2]) / (2*dx)

    u_x_t2[0]    = (
                        u_x_t[0] \
                        + dt / l__t1 * u_x_t[0] * dudx 
                        - dt / l__t1 * kappa * (dDdx + dhdx)
                    ) / (1 + Lambda[0] * dt)

    dudx = -(-3 * u_x_t[-1] + 4 * u_x_t[-2] - u_x_t[-3]) / (2*dx)
    dDdx = -(-3 * D_x_t[-1] + 4 * D_x_t[-2] - D_x_t[-3]) / (2*dx)
    dhdx = -(- 3 * h_x[-1] + 4 * h_x[-2] - h_x[-3]) / (2*dx)

    u_x_t2[-1]   = (
                        u_x_t[-1] \
                        + dt / l__t1 * ((l__t2 - l__t1) / dt - u_x_t[-1]) * dudx # this is zero???
                        - dt / l__t1 * kappa * (dDdx + dhdx)
                    ) / (1 + Lambda[-1] * dt)
    

    dudx = (u_x_t[2:] - u_x_t[:-2]) / (2*dx)
    dDdx = (D_x_t[2:] - D_x_t[:-2]) / (2*dx)
    dhdx = (h_x[2:] - h_x[:-2]) / (2*dx)

    u_x_t2[1:-1] = (
                        (u_x_t[2:] + u_x_t[:-2]) / 2 \
                        + dt / l__t1 * (x_x[1:-1] * (l__t2 - l__t1) / dt - u_x_t[1:-1]) * dudx \
                        - dt / l__t1 * kappa * (dDdx + dhdx)
                    ) / (1 + Lambda[1:-1] * dt)
    
    return u_x_t2

# implicit, linear friction term
def update_u2(l__t1, l__t2, u_x_t, x_x, h_x, D_x_t, D_x_t2, dt, dx, r, h0, kappa):

    def residual(u):
        res = np.zeros(u.shape)

        dudx = (-3 * u[0] + 4 * u[1] - u[2]) / (2*dx)
        dDdx = (-3 * D_x_t[0] + 4 * D_x_t[1] - D_x_t[2]) / (2*dx)
        dhdx = (-3 * h_x[0] + 4 * h_x[1] - h_x[2]) / (2*dx)

        res[0]    = (
                            u_x_t[0] \
                            + dt / l__t1 * u[0] * dudx 
                            - dt / l__t1 * kappa * (dDdx + dhdx)
                            - dt * r / (D_x_t2[0] + h0) * u[0]
                        ) - u[0]

        dudx = -(-3 * u[-1] + 4 * u[-2] - u[-3]) / (2*dx)
        dDdx = -(-3 * D_x_t[-1] + 4 * D_x_t[-2] - D_x_t[-3]) / (2*dx)
        dhdx = -(- 3 * h_x[-1] + 4 * h_x[-2] - h_x[-3]) / (2*dx)

        res[-1]   = (
                            u_x_t[-1] \
                            + dt / l__t1 * ((l__t2 - l__t1) / dt - u[-1]) * dudx # this is zero???
                            - dt / l__t1 * kappa * (dDdx + dhdx)
                            - dt * r / (D_x_t2[-1] + h0) * u[-1]
                        ) - u[-1]
        
        dudx = (u[2:] - u[:-2]) / (2*dx)
        dDdx = (D_x_t[2:] - D_x_t[:-2]) / (2*dx)
        dhdx = (h_x[2:] - h_x[:-2]) / (2*dx)
        
        res[1:-1] = (
                            (u_x_t[2:] + u_x_t[:-2]) / 2 \
                            + dt / l__t1 * (x_x[1:-1] * (l__t2 - l__t1) / dt - u[1:-1]) * dudx \
                            - dt / l__t1 * kappa * (dDdx + dhdx) \
                            - dt * r / (D_x_t2[1:-1] + h0) * u[1:-1]
                        ) - u[1:-1]
        
        return res
    
    def construct_jac(u):
        jac = np.zeros((nx + 2, nx + 2))


        # relations with u_{j-1} and u_{j+1}
        cst = dt / l__t1 * (x_x[1:-1] * (l__t2 - l__t1) / dt - u[1:-1]) / 2 / dx

        upper_diag = np.diag(cst, 1)
        lower_diag = np.daig(-cst, 1)


        dudx = (u[2:] - u[:-2]) / (2*dx)

        cst = ( 
            - dt / l__t1 * dudx
            - dt * c_d * np.sign(u_x_t[1:-1]) /  2 / (D_x_t2[1:-1] + h0)
        )

        jac[1:-1, :] = [] 

        return jac

    guess = u_x_t
    u_x_t2 = scipy.optimize.root(residual, guess).x

    return u_x_t2

# explicit, quadratic friction term
def update_u3(l__t1, l__t2, u_x_t, x_x, h_x, D_x_t, D_x_t2, dt, dx, c_d, h0, kappa):

    # two options to compute lambda
    # Lambda_start = c_d * np.abs(u_x_t[0]) / (D_x_t2[0] + h0)**(4/3)
    # Lambda_middle = c_d * np.abs(u_x_t[:-2] + u_x_t[2:]) / (D_x_t2[1:-1] + h0)**(4/3)
    # Lambda_end = c_d * np.abs(u_x_t[-1]) / (D_x_t2[-1] + h0)**(4/3)
    # Lambda = np.array([Lambda_start] + Lambda_middle.tolist() + [Lambda_end])

    Lambda = c_d * np.abs(u_x_t) / (D_x_t2 + h0)**(4/3)


    u_x_t2 = np.zeros(u_x_t.shape)

    dudx = (-3 * u_x_t[0] + 4 * u_x_t[1] - u_x_t[2]) / (2*dx)
    dDdx = (-3 * D_x_t[0] + 4 * D_x_t[1] - D_x_t[2]) / (2*dx)
    dhdx = (-3 * h_x[0] + 4 * h_x[1] - h_x[2]) / (2*dx)

    u_x_t2[0]    = (
                        u_x_t[0] \
                        + dt / l__t1 * u_x_t[0] * dudx 
                        - dt / l__t1 * kappa * (dDdx + dhdx)
                    ) / (1 + Lambda[0] * dt)

    dudx = -(-3 * u_x_t[-1] + 4 * u_x_t[-2] - u_x_t[-3]) / (2*dx)
    dDdx = -(-3 * D_x_t[-1] + 4 * D_x_t[-2] - D_x_t[-3]) / (2*dx)
    dhdx = -(- 3 * h_x[-1] + 4 * h_x[-2] - h_x[-3]) / (2*dx)

    u_x_t2[-1]   = (
                        u_x_t[-1] \
                        + dt / l__t1 * ((l__t2 - l__t1) / dt - u_x_t[-1]) * dudx # this is zero???
                        - dt / l__t1 * kappa * (dDdx + dhdx)
                    ) / (1 + Lambda[-1] * dt)
    

    dudx = (u_x_t[2:] - u_x_t[:-2]) / (2*dx)
    dDdx = (D_x_t[2:] - D_x_t[:-2]) / (2*dx)
    dhdx = (h_x[2:] - h_x[:-2]) / (2*dx)

    u_x_t2[1:-1] = (
                        (u_x_t[2:] + u_x_t[:-2]) / 2 \
                        + dt / l__t1 * (x_x[1:-1] * (l__t2 - l__t1) / dt - u_x_t[1:-1]) * dudx \
                        - dt / l__t1 * kappa * (dDdx + dhdx)
                    ) / (1 + Lambda[1:-1] * dt)
    
    return u_x_t2

# implicit, quadratic friction term
def update_u4(l__t1, l__t2, u_x_t, x_x, h_x, D_x_t, D_x_t2, dt, dx, c_d, h0, kappa):

    def residual(u):
        res = np.zeros(u.shape)

        dudx = (-3 * u[0] + 4 * u[1] - u[2]) / (2*dx)
        dDdx = (-3 * D_x_t[0] + 4 * D_x_t[1] - D_x_t[2]) / (2*dx)
        dhdx = (-3 * h_x[0] + 4 * h_x[1] - h_x[2]) / (2*dx)

        res[0]    = (
                            u_x_t[0] \
                            + dt / l__t1 * u[0] * dudx 
                            - dt / l__t1 * kappa * (dDdx + dhdx)
                            - dt * c_d * np.sign(u_x_t[0]) / (D_x_t2[0] + h0) * u[0]**2
                        ) - u[0]

        dudx = -(-3 * u[-1] + 4 * u[-2] - u[-3]) / (2*dx)
        dDdx = -(-3 * D_x_t[-1] + 4 * D_x_t[-2] - D_x_t[-3]) / (2*dx)
        dhdx = -(- 3 * h_x[-1] + 4 * h_x[-2] - h_x[-3]) / (2*dx)

        res[-1]   = (
                            u_x_t[-1] \
                            + dt / l__t1 * ((l__t2 - l__t1) / dt - u[-1]) * dudx # this is zero???
                            - dt / l__t1 * kappa * (dDdx + dhdx)
                            - dt * c_d * np.sign(u_x_t[-1]) / (D_x_t2[-1] + h0) * u[-1]**2
                        ) - u[-1]
        
        dudx = (u[2:] - u[:-2]) / (2*dx)
        dDdx = (D_x_t[2:] - D_x_t[:-2]) / (2*dx)
        dhdx = (h_x[2:] - h_x[:-2]) / (2*dx)
        
        res[1:-1] = (
                            (u_x_t[2:] + u_x_t[:-2]) / 2 \
                            + dt / l__t1 * (x_x[1:-1] * (l__t2 - l__t1) / dt - u[1:-1]) * dudx \
                            - dt / l__t1 * kappa * (dDdx + dhdx) \
                            - dt * c_d * np.sign(u_x_t[1:-1]) / (D_x_t2[1:-1] + h0) * u[1:-1]**2
                        ) - u[1:-1]
        
        return res
    
    def construct_jac(u):
        jac = np.zeros((nx + 2, nx + 2))


        # relations with u_{j-1} and u_{j+1}
        cst = dt / l__t1 * (x_x[1:-1] * (l__t2 - l__t1) / dt - u[1:-1]) / 2 / dx

        upper_diag = np.diag(cst, 1)
        lower_diag = np.daig(-cst, 1)


        dudx = (u[2:] - u[:-2]) / (2*dx)

        cst = ( 
            - dt / l__t1 * dudx
            - dt * c_d * np.sign(u_x_t[1:-1]) /  2 / (D_x_t2[1:-1] + h0)
        )

        jac[1:-1, :] = [] 

        return jac

    guess = u_x_t
    u_x_t2 = scipy.optimize.root(residual, guess).x
    
    return u_x_t2


def solve_pde(A, kappa, H, friction_parameter, h0, nx, nr_periods, update_u):

    # numerical precision
    dx = 1 / nx    
    dt = 0.9 / np.sqrt(kappa) * dx # CFL condition
    nt = int(nr_periods * 2 * pi / dt)

    # store the results here
    t = np.linspace(0, dt * nt, nt + 1)
    x_x = np.linspace(-dx, 1 + dx, nx + 2)
    l_t_ = np.zeros(nt + 1)
    D_xt_ = np.zeros((nx + 2, nt + 1))
    u_xt_ = np.zeros((nx + 2, nt + 1))
    
    # initial conditions
    D_xt_[:, 0] = (H + A) / H - x_x # since x_x = h_x
    l_t_[0] = 1 # 2e4 / 1.9e4


    for timestep in range(nt):
        if timestep % 1000 == 0:
            print(timestep)
        l_t_[timestep + 1] = update_l(l_t_[timestep], u_xt_[:, timestep], dt)
        D_xt_[:, timestep + 1] = update_D(l_t_[timestep], l_t_[timestep + 1], u_xt_[:, timestep], x_x, D_xt_[:, timestep], dt, dx, timestep * dt, A, H)
        u_xt_[:, timestep + 1] = update_u(l_t_[timestep], l_t_[timestep + 1], u_xt_[:, timestep], x_x, x_x * l_t_[timestep], D_xt_[:, timestep], D_xt_[:, timestep + 1], dt, dx, friction_parameter, h0, kappa)


    return t, x_x, l_t_, D_xt_, u_xt_


if __name__ == "__main__":
    nx, nr_periods = 500, 3
    t, x_x, l_t_, D_xt_, u_xt_ = solve_pde(A, kappa, H, r, h0, nx, nr_periods, update_u1)
    plt.plot(t, l_t_)

    # t, x_x, l_t_, D_xt_, u_xt_ = solve_pde(A, kappa, H, r, h0, 100, nr_periods, update_u2)
    # plt.plot(t, l_t_)

    n = 0.03
    rho = 1000
    g = 9.81
    c_d = g * n**2
    print(c_d)
    t, x_x, l_t_, D_xt_, u_xt_ = solve_pde(A, kappa, H, c_d, h0, nx, nr_periods, update_u3)
    plt.plot(t, l_t_)

    c_d = 0.4
    t, x_x, l_t_, D_xt_, u_xt_ = solve_pde(A, kappa, H, c_d, h0, nx, nr_periods, update_u3)
    plt.plot(t, l_t_)

    plt.show()
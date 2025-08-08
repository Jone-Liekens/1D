
# region Imports
import numpy as np
from numpy import sin, cos, tan, atan, cosh, sinh, tanh, abs, linspace, min, max, argmin, argmax, pi, mean, exp, sqrt
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import os
# endregion

def heatmap(T_mesh, X_mesh, V_xt, title = "Something"):
    fig, ax = plt.subplots()

    # print(T_mesh.shape, X_mesh.shape, V_xt.shape)

    im = ax.pcolormesh(T_mesh, X_mesh, V_xt, cmap='inferno', shading='nearest')
    fig.colorbar(im, ax=ax)#, label='Value of $u$')

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Spatial Dimension (x)')
    ax.set_title(title)

    plt.show()

def spatial_aggregates(x, v_xt, title = "Something"):

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].plot(x, mean(v_xt, axis=1))
    axs[0].set_title('Mean ' + title)
    axs[0].set_xlabel('Space (x)')

    axs[1].plot(x, max(v_xt, axis=1))
    axs[1].set_title('Max ' + title)
    axs[1].set_xlabel('Space (x)')

    axs[2].plot(x, min(v_xt, axis=1))
    axs[2].set_title('Min ' + title)
    axs[2].set_xlabel('Space (x)')

    axs[3].plot(x, min(abs(v_xt), axis=1))
    axs[3].set_title('Min abs ' + title)
    axs[3].set_xlabel('Space (x)')

    plt.show()


def t_snapshots(x, t, v_xt, title="Something", n_snapshots=7):

    for n in range(n_snapshots):
        i = (n * len(t)) // n_snapshots
        plt.plot(x, v_xt[:, i])

    plt.legend(["t = " + str(t[ (n * len(t)) // n_snapshots])[:6] for n in range(n_snapshots)])
    plt.title(title)
    plt.xlabel('Space (x)')
    plt.show()


# this is not correct: x should be multiplied by l(t)
def x_snapshots(x, t, v_xt, title="Something", n_snapshots=7):
    x = x[3:-3]
    for n in range(n_snapshots):
        i = (n * len(x)) // n_snapshots
        plt.plot(t, v_xt[i, :])

    plt.legend(["x = " + str(x[ (n * len(x)) // n_snapshots])[:6] for n in range(n_snapshots)])
    plt.title(title)
    plt.xlabel('Time (i)')
    plt.show()


def harmonic_components(x, r, c, s, title="Harmonics"):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))


    axs[0].plot(x, r)
    axs[0].set_title('Residual')
    axs[0].set_xlabel('Space (x)')

    axs[1].plot(x, c)
    axs[1].set_title('Cosine')
    axs[1].set_xlabel('Space (x)')

    axs[2].plot(x, s)
    axs[2].set_title('Sine')
    axs[2].set_xlabel('Space (x)')

    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.show()



def pickle_dump(object, location):
    try:
        with open(location, 'wb') as f: # 'wb' means write in binary mode
            pickle.dump(object, f)
        print(f"Object successfully pickled to '{location}'")
    except Exception as e:
        print(f"An error occurred during pickling: {e}")


def pickle_load(location):
    try:
        if os.path.exists(location):
            with open(location, 'rb') as f: # 'rb' means read in binary mode
                unpickled_data = pickle.load(f)
            print(f"Object successfully unpickled from '{location}'")
        else:
            print(f"Pickle file '{location}' not found.")
    except Exception as e:
        print(f"An error occurred during unpickling: {e}")
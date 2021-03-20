#!/usr/bin/env python
# coding: utf-8


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm
import matplotlib.colors

from matplotlib import animation, rc
# from IPython.display import HTML

import numpy as np
from .momentum import Momentum
# from .nag import NAG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algoritm',
        help='type of the algoritm you want to visualize ;',
        type=str)
    parser.add_argument(
        '--gamma',
        help='number of epochs;',
        type=float,
        default=0.9)
    parser.add_argument(
        '--local_epochs',
        help='number of local epochs;',
        type=int,
        default=1)
    parser.add_argument(
        '--gamma',
        type=int)
    parser.add_argument(
        '--degree',
        type=int,
        default=3)
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=1234,
        required=False)
    return parser.parse_args()


def visualize_3d(neuron_class):

    W = np.linspace(w_min, w_max, 256)
    b = np.linspace(b_min, b_max, 256)
    WW, BB = np.meshgrid(W, b)
    Z = neuron_class.error(X, Y, WW, BB)

    fig = plt.figure(dpi=200)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(WW, BB, Z, rstride=3, cstride=3,
                           alpha=0.5, cmap=cm.magma,
                           linewidth=0, antialiased=False)
    cset = ax.contourf(WW, BB, Z, 25, zdir='z',
                       offset=-1, alpha=0.6, cmap=cm.magma
                       )
    ax.set_xlabel('w')
    ax.set_xlim(w_min - 1, w_max + 1)
    ax.set_ylabel('b')
    ax.set_ylim(b_min - 1, b_max + 1)
    ax.set_zlabel('error')
    ax.set_zlim(-1, np.max(Z))
    ax.view_init(elev=25, azim=-20)  # azim = -20
    ax.dist = 12

    i = 0
    line1, = ax.plot(np.asarray(neuron_class.weight_h[:i + 1]),
                     np.asarray(neuron_class.bias_h[:i + 1]),
                     np.asarray(neuron_class.error_h[:i + 1]), color='black', marker='.'
                     )
    line2, = ax.plot(np.asarray(neuron_class.weight_h[:i + 1]),
                     np.asarray(neuron_class.bias_h[:i + 1]),
                     np.zeros(i + 1) - 1, color='red', marker='.')
    title = ax.set_title('Epoch 0')
    anim = animation.FuncAnimation(fig, func=plot_animate_3d,
                                   frames=animation_frames, fargs=(
                                       neuron_class, line1, line2, title)
                                   )
    rc('animation', html='jshtml')
    plt.show()


def plot_animate_3d(i, neuron_class, line1, line2, title):
    i = int(i * (epochs / animation_frames))

    line1.set_data(np.asarray(
        neuron_class.weight_h[:i + 1]), np.asarray(neuron_class.bias_h[:i + 1])
    )
    line1.set_3d_properties(np.asarray(neuron_class.error_h[:i + 1]))
    line2.set_data(
        np.asarray(neuron_class.weight_h[:i + 1]),
        np.asarray(neuron_class.bias_h[:i + 1])
    )
    line2.set_3d_properties(np.asarray(np.zeros(i + 1) - 1))
    title.set_text('Epoch: {: d}, Error: {:.4f}'.format(
        i, neuron_class.error_h[i]))
    return line1, line2, title


def plot_animate_2d(i, neuron_class, line, title):
    i = int(i * (epochs / animation_frames))
    line.set_data(neuron_class.weight_h[:i + 1], neuron_class.bias_h[:i + 1])
    title.set_text('Epoch: {: d}, Error: {:.4f}'.format(
        i, neuron_class.error_h[i]))
    return line, title


def visualize_2d(neuron_class):
    W = np.linspace(w_min, w_max, 256)
    b = np.linspace(b_min, b_max, 256)
    WW, BB = np.meshgrid(W, b)
    Z = neuron_class.error(X, Y, WW, BB)

    fig = plt.figure(dpi=300)
    ax = plt.subplot(111)
    ax.set_xlabel('w')
    ax.set_xlim(w_min - 1, w_max + 1)
    ax.set_ylabel('b')
    ax.set_ylim(b_min - 1, b_max + 1)
    title = ax.set_title('Epoch 0')
    cset = plt.contourf(WW, BB, Z, 25, alpha=0.6, cmap=cm.bwr)
    i = 0
    line, = ax.plot(
        neuron_class.weight_h[:i + 1],
        neuron_class.bias_h[:i + 1],
        color='black', marker='.'
    )
    anim = animation.FuncAnimation(
        fig, func=plot_animate_2d, frames=animation_frames, fargs=(
            neuron_class, line, title))
    rc('animation', html='jshtml')
    plt.show()


if __name__ == '__main__':

    X = np.asarray([3.5, 0.35, 3.2, -2.0, 1.5, -0.5])
    Y = np.asarray([0.5, 0.52, 0.56, 0.51, 0.12, 0.35])
    print("X : ", X)
    print("Y : ", Y)
    w_init = -2
    b_init = -2
    w_min = -10
    w_max = 10
    b_min = -10
    b_max = 10
    epochs = 100
    mini_batch_size = 6
    gamma = 0.9
    eta = 1
    animation_frames = 120
    plot_2d = True
    plot_3d = True

    momentum = Momentum(w_init, b_init)
    plt.style.use("seaborn")
    momentum.fit(X, Y, epochs=epochs, eta=eta, gamma=gamma)
    plt.plot(momentum.error_h, 'r')
    plt.plot(momentum.weight_h, 'b')
    plt.plot(momentum.bias_h, 'g')
    plt.legend(["Error", "Weight", "Bias"])
    # w_diff = [t - s for t, s in zip(sn.w_h, sn.w_h[1:])]
    # b_diff = [t - s for t, s in zip(sn.b_h, sn.b_h[1:])]
    # plt.plot(w_diff, 'b--')
    # plt.plot(b_diff, 'g--')
    plt.title("Variation of Parameters and Error in  GD")
    plt.show()
    visualize_2d(momentum)

# nag = NAG(w_init, b_init)
# plt.style.use("seaborn")
# nag.fit(X, Y, epochs=epochs, eta=eta, gamma=gamma)
# plt.plot(nag.error_h, 'r')
# plt.plot(nag.weight_h, 'b')
# plt.plot(nag.bias_h, 'g')
# plt.legend(["Error", "Weight", "Bias"])
# # w_diff = [t - s for t, s in zip(sn.w_h, sn.w_h[1:])]
# # b_diff = [t - s for t, s in zip(sn.b_h, sn.b_h[1:])]
# # plt.plot(w_diff, 'b--')
# # plt.plot(b_diff, 'g--')
# plt.title("Variation of Parameters and Error in  GD")
# plt.show()]


# import sys

# if __name__ == "__main__":
#     algoritm
#     print sys.argv[0]  # prints python_script.py
#     print sys.argv[1]  # prints var1
#     print sys.argv[2]  # prints var2

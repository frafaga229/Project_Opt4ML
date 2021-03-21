from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors

from matplotlib import animation, rc
# from IPython.display import HTML

import numpy as np
from .momentum import Momentum
from .nag import NAG


X = np.asarray([3.5, 0.35, 3.2, -2.0, 1.5, -0.5, -0.25])
Y = np.asarray([0.5, 0.52, 0.56, 0.51, 0.12, 0.35, 0.75])
w_init = -6
b_init = 5
w_min = -7
w_max = 5
b_min = -7
b_max = 5
epochs = 200
gamma = 0.9
eta = 0.8
eps = 1e-8
animation_frames = 40


def visualize_3d(neuron_class1, neuron_class2, fig):

    W = np.linspace(w_min, w_max, 256)
    b = np.linspace(b_min, b_max, 256)
    WW, BB = np.meshgrid(W, b)
    Z = neuron_class1.error(X, Y, WW, BB)

    # fig = plt.figure(dpi=200)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(WW, BB, Z, rstride=2, cstride=2,
                            alpha=0.3, cmap=cm.bwr,
                            linewidth=0, antialiased=False)
    fig.colorbar(surf, orientation="horizontal", aspect=5)
    ax1.set_xlabel('w')
    ax1.set_xlim(w_min - 1, w_max + 1)
    ax1.set_ylabel('b')
    ax1.set_ylim(b_min - 1, b_max + 1)
    ax1.set_zlabel('error')
    ax1.set_zlim(-1, np.max(Z))
    ax1.view_init(elev=25, azim=-10)  # azim = -20
    ax1.dist = 8

    i = 0
    line11, = ax1.plot(np.asarray(neuron_class1.weight_h[:i + 1]),
                       np.asarray(neuron_class1.bias_h[:i + 1]),
                       np.asarray(neuron_class1.error_h[:i + 1]), color='lightgreen', marker='o', linewidth=2, markersize=3, label='Momentum'
                       )
    line12, = ax1.plot(np.asarray(neuron_class2.weight_h[:i + 1]),
                       np.asarray(neuron_class2.bias_h[:i + 1]),
                       np.asarray(neuron_class2.error_h[:i + 1]), color='orange', marker='o', linewidth=2, markersize=3, label="NAG"
                       )
    ax1.legend(facecolor='white', frameon=True, prop={'size': 18})
    title = fig.suptitle('Epoch 0')
    return line11, line12, title


def plot_animate_3d(j, neuron_class1, neuron_class2, line11, line12, title):
    i = int(j * (epochs / animation_frames))
    line11.set_data(np.asarray(
        neuron_class1.weight_h[:i + 1]), np.asarray(neuron_class1.bias_h[:i + 1])
    )
    line11.set_3d_properties(np.asarray(neuron_class1.error_h[:i + 1]))
    line11.set_3d_properties(np.asarray(neuron_class1.error_h[:i + 1]))

    line12.set_data(np.asarray(
        neuron_class2.weight_h[:i + 1]), np.asarray(neuron_class2.bias_h[:i + 1])
    )
    line12.set_3d_properties(np.asarray(neuron_class2.error_h[:i + 1]))
    line12.set_3d_properties(np.asarray(neuron_class2.error_h[:i + 1]))
    title.set_text('Epoch: {: d}, Error-momentum: {:.4f}, , Error-NAG: {:.4f}'.format(
        i, neuron_class1.error_h[i], neuron_class2.error_h[i]))
    return line11, line12, title


def plot_animate_2d(k, neuron_class1, neuron_class2, line21, line22, title):
    i = int(k * (epochs / animation_frames))
    line21.set_data(
        neuron_class1.weight_h[:i + 1], neuron_class1.bias_h[:i + 1]
    )
    line22.set_data(
        neuron_class2.weight_h[:i + 1], neuron_class2.bias_h[:i + 1]
    )
    title.set_text('Epoch: {: d}, Error-momentum: {:.4f}, , Error-NAG: {:.4f}'.format(
        i, neuron_class1.error_h[i], neuron_class2.error_h[i]))
    return line21, line22, title


def updateALL(i, neuron_class1, neuron_class2, line11, line12, line21, line22, title):
    a = plot_animate_3d(i, neuron_class1, neuron_class2, line11, line12, title)
    b = plot_animate_2d(i, neuron_class1, neuron_class2, line21, line22, title)
    return a + b


def visualize_2d(neuron_class1, neuron_class2, fig):
    W = np.linspace(w_min, w_max, 256)
    b = np.linspace(b_min, b_max, 256)
    WW, BB = np.meshgrid(W, b)
    Z = neuron_class1.error(X, Y, WW, BB)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('w')
    ax2.set_xlim(w_min - 1, w_max + 1)
    ax2.set_ylabel('b')
    ax2.set_ylim(b_min - 1, b_max + 1)
    title = fig.suptitle('Epoch 0')
    cset = plt.contourf(WW, BB, Z, 50, alpha=0.5, cmap=cm.bwr, extend='both')
    i = 0
    line21, = ax2.plot(
        neuron_class1.weight_h[:i + 1],
        neuron_class1.bias_h[:i + 1],
        color='lightgreen', marker='o', linewidth=2, markersize=3, label='Momentum'
    )
    line22, = ax2.plot(
        neuron_class2.weight_h[:i + 1],
        neuron_class2.bias_h[:i + 1],
        color='orange', marker='o', linewidth=2, markersize=3, label="NAG"
    )
    ax2.legend(facecolor='white', frameon=True, prop={'size': 18})
    return line21, line22, title


def start_update_visualization(fig, neuron_class1, neuron_class2, line11, line12, line21, line22, title):
    anim = animation.FuncAnimation(
        fig, func=updateALL, frames=animation_frames, repeat=False, fargs=(
            neuron_class1, neuron_class2, line11, line12, line21, line22, title
        ))
    # rc('animation', html='jshtml')
    plt.show()


def toy_example(epochs_, gamma_, eta_):
    global epochs, gamma, eta
    epochs = epochs_
    gamma = gamma_
    eta = eta_

    print("X : ", X)
    print("Y : ", Y)

    momentum = Momentum(w_init, b_init)
    plt.style.use("seaborn")
    momentum.fit(X, Y, epochs=epochs, eta=eta, gamma=gamma)
    plt.plot(momentum.error_h, 'r')
    plt.plot(momentum.weight_h, 'b')
    plt.plot(momentum.bias_h, 'g')
    plt.legend(["Error", "Weight", "Bias"], frameon=True, prop={'size': 18})
    # w_diff = [t - s for t, s in zip(sn.w_h, sn.w_h[1:])]
    # b_diff = [t - s for t, s in zip(sn.b_h, sn.b_h[1:])]
    # plt.plot(w_diff, 'b--')
    # plt.plot(b_diff, 'g--')
    plt.title("Variation of Parameters and Error in Momentum")
    plt.show()

    nag = NAG(w_init, b_init)
    plt.style.use("seaborn")
    nag.fit(X, Y, epochs=epochs, eta=eta, gamma=gamma)
    plt.plot(nag.error_h, 'r')
    plt.plot(nag.weight_h, 'b')
    plt.plot(nag.bias_h, 'g')
    plt.legend(["Error", "Weight", "Bias"], frameon=True, prop={'size': 18})
    # w_diff = [t - s for t, s in zip(sn.w_h, sn.w_h[1:])]
    # b_diff = [t - s for t, s in zip(sn.b_h, sn.b_h[1:])]
    # plt.plot(w_diff, 'b--')
    # plt.plot(b_diff, 'g--')
    plt.title("Variation of Parameters and Error in NAG")
    plt.show()

    fig = plt.figure(figsize=(40, 20))
    line11, line12, title = visualize_3d(momentum, nag, fig)
    line21, line22, title = visualize_2d(momentum, nag, fig)
    start_update_visualization(fig, momentum, nag, line11,
                               line12, line21, line22, title
                               )

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors


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

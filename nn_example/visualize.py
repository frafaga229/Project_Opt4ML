from .momentum_nn import MomentumNetwork
from .sgd_nn import SGDNetwork
from .nag_nn import NAGNetwork
import matplotlib.pyplot as plt


def visualize_nn(W1, W2, X_train, y_OH_train, epochs, gamma, eta):

    # Create model neuron network with method SGD without momentum
    model_sgd = SGDNetwork(W1, W2)
    model_sgd.fit(X_train, y_OH_train, epochs=epochs, eta=eta,
                  algo="Momentum", gamma=gamma, display_loss=True)
    # Create model neuron network with momentum method
    model_moment = MomentumNetwork(W1, W2)
    model_moment.fit(X_train, y_OH_train, epochs=epochs, eta=eta,
                     algo="Momentum", gamma=gamma, display_loss=True)
    # Create model neuron network with Nesterov's accelerated gradient method
    model_nag = NAGNetwork(W1, W2)
    model_nag.fit(X_train, y_OH_train, epochs=epochs, eta=eta,
                  algo="Momentum", gamma=gamma, display_loss=True)
    # Plot from loss values of 3 models
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(model_sgd.loss.values(), '-v',
             markersize=5, label='SGD - loss')
    plt.plot(model_moment.loss.values(), '-x',
             markersize=5, label='Momentum - loss')
    plt.plot(model_nag.loss.values(), '-o',
             markersize=5, label='NAG - loss')
    plt.legend()
    plt.title("Comparing Loss of 3 method SGD, Momentum and NAG")
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(model_sgd.acc.values(), '-v',
             markersize=5, label='SGD - acc')
    plt.plot(model_moment.acc.values(), '-x',
             markersize=5, label='Momentum - acc')
    plt.plot(model_nag.acc.values(), '-o',
             markersize=5, label='NAG - acc')
    plt.title("Comparing Accuracy of 3 method SGD, Momentum and NAG")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

from .momentum_nn import MomentumNetwork
from .sgd_nn import SGDNetwork
from .nag_nn import NAGNetwork
import matplotlib.pyplot as plt


def visualize_nn(W1, W2, X_train, y_OH_train, epochs, gamma, eta):

    model_sgd = SGDNetwork(W1, W2)
    model_sgd.fit(X_train, y_OH_train, epochs=epochs, eta=eta,
                  algo="Momentum", gamma=gamma, display_loss=True)

    model_moment = MomentumNetwork(W1, W2)
    model_moment.fit(X_train, y_OH_train, epochs=epochs, eta=eta,
                     algo="Momentum", gamma=gamma, display_loss=True)

    model_nag = NAGNetwork(W1, W2)
    model_nag.fit(X_train, y_OH_train, epochs=epochs, eta=eta,
                  algo="Momentum", gamma=gamma, display_loss=True)

    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(model_sgd.loss.values(), '-v',
             markersize=5, label='SGD - loss')
    plt.plot(model_moment.loss.values(), '-x',
             markersize=5, label='Momentum - loss')
    plt.legend()
    plt.plot(model_nag.loss.values(), '-o',
             markersize=5, label='NAG - loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(model_sgd.acc.values(), '-v',
             markersize=5, label='SGD - acc')
    plt.plot(model_moment.acc.values(), '-x',
             markersize=5, label='Momentum - acc')
    plt.legend()
    plt.plot(model_nag.acc.values(), '-o',
             markersize=5, label='NAG - acc')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
import argparse
from .visualize import visualize_nn


from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gamma',
        help='number of epochs;',
        type=float,
        default=0.9)
    parser.add_argument(
        '--epochs',
        help='number of epochs;',
        type=int,
        default=200)
    parser.add_argument(
        '--eta',
        type=float,
        default=0.8)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["orange", "purple", "green"])

    np.random.seed(0)
    data, labels = make_blobs(n_samples=1000, centers=4,
                              n_features=2, random_state=0)

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
    plt.title("Example data for classification (4 labels)")
    plt.show()

    X_train, X_val, Y_train, Y_val = train_test_split(
        data, labels, stratify=labels, random_state=0
    )
    print(X_train.shape, X_val.shape, labels.shape)

    enc = OneHotEncoder()
    # 0 -> (1, 0, 0, 0), 1 -> (0, 1, 0, 0), 2 -> (0, 0, 1, 0), 3 -> (0, 0, 0, 1)
    y_OH_train = enc.fit_transform(np.expand_dims(Y_train, 1)).toarray()
    y_OH_val = enc.fit_transform(np.expand_dims(Y_val, 1)).toarray()

    W1 = np.random.randn(2, 2)
    W2 = np.random.randn(2, 4)

    visualize_nn(W1, W2, X_train, y_OH_train,
                 args.epochs, args.gamma, args.eta)

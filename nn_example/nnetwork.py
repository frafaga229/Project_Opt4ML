
import numpy as np


class NNetwork:

    def __init__(self, W1, W2):

        self.params = {}
        self.params["W1"] = W1.copy()
        self.params["W2"] = W2.copy()
        self.params["B1"] = np.zeros((1, 2))
        self.params["B2"] = np.zeros((1, 4))
        self.num_layers = 2
        self.gradients = {}
        self.update_params = {}
        self.prev_update_params = {}
        self.loss = {}
        self.acc = {}
        for i in range(1, self.num_layers + 1):
            self.update_params["v_w" + str(i)] = 0
            self.update_params["v_b" + str(i)] = 0
            self.update_params["m_b" + str(i)] = 0
            self.update_params["m_w" + str(i)] = 0
            self.prev_update_params["v_w" + str(i)] = 0
            self.prev_update_params["v_b" + str(i)] = 0

    def forward_activation(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def grad_activation(self, X):
        return X * (1 - X)

    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

    def forward_pass(self, X, params=None):
        if params is None:
            params = self.params
        self.A1 = np.matmul(X, params["W1"]) + \
            params["B1"]  # (N, 2) * (2, 2) -> (N, 2)
        self.H1 = self.forward_activation(self.A1)  # (N, 2)
        # (N, 2) * (2, 4) -> (N, 4)
        self.A2 = np.matmul(self.H1, params["W2"]) + params["B2"]
        self.H2 = self.softmax(self.A2)  # (N, 4)
        return self.H2

    def grad(self, X, Y, params=None):
        if params is None:
            params = self.params

        self.forward_pass(X, params)
        self.gradients["dA2"] = self.H2 - Y  # (N, 4) - (N, 4) -> (N, 4)
        self.gradients["dW2"] = np.matmul(
            self.H1.T, self.gradients["dA2"])  # (2, N) * (N, 4) -> (2, 4)
        self.gradients["dB2"] = np.sum(
            self.gradients["dA2"], axis=0).reshape(1, -1)  # (N, 4) -> (1, 4)
        self.gradients["dH1"] = np.matmul(
            self.gradients["dA2"], params["W2"].T)  # (N, 4) * (4, 2) -> (N, 2)
        self.gradients["dA1"] = np.multiply(self.gradients["dH1"], self.grad_activation(
            self.H1))  # (N, 2) .* (N, 2) -> (N, 2)
        self.gradients["dW1"] = np.matmul(
            X.T, self.gradients["dA1"])  # (2, N) * (N, 2) -> (2, 2)
        self.gradients["dB1"] = np.sum(
            self.gradients["dA1"], axis=0).reshape(1, -1)  # (N, 2) -> (1, 2)

    def predict(self, X):
        Y_pred = self.forward_pass(X)
        return np.array(Y_pred).squeeze()

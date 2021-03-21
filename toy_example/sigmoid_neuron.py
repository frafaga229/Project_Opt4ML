import numpy as np

# This is a abstract class sigmoid
class SN:

    def __init__(self, weight_init, bias_init):
        self.weight = weight_init
        self.bias = bias_init
        self.weight_h = []
        self.bias_h = []
        self.error_h = []
    # This function to compute sigmoid from weight and bias
    def sigmoid(self, x, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return 1. / (1. + np.exp(-(weight * x + bias)))

    # This function is compute error
    def error(self, X, Y, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        err = 0
        for x, y in zip(X, Y):
            # We use MSE error to compute error in this case
            err += 0.5 * (self.sigmoid(x, weight, bias) - y) ** 2
        return err

    # This function is compute gradient on weight parameter
    def grad_weight(self, x, y, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        y_pred = self.sigmoid(x, weight, bias)
        return (y_pred - y) * y_pred * (1 - y_pred) * x

    # This function is compute gradient on bias parameter
    def grad_bias(self, x, y, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        y_pred = self.sigmoid(x, weight, bias)
        return (y_pred - y) * y_pred * (1 - y_pred)

    # This function is to add values to 3 list to visualize
    def append_log(self):
        self.weight_h.append(self.weight)
        self.bias_h.append(self.bias)
        self.error_h.append(self.error(self.X, self.Y))


from .sigmoid_neuron import SN


class NAG(SN):

    def __init__(self, weight_init, bias_init):
        self.weight = weight_init
        self.bias = bias_init
        self.weight_h = []
        self.bias_h = []
        self.error_h = []

    def fit(self, X, Y,
            epochs=100, eta=0.01, gamma=0.9, mini_batch_size=100, eps=1e-8,
            beta=0.9, beta1=0.9, beta2=0.9):

        self.weight_h = []
        self.bias_h = []
        self.error_h = []
        self.X = X
        self.Y = Y
        v_weight, v_bias = 0, 0

        for i in range(epochs):
            dweight, dbias = 0, 0
            v_weight = gamma * v_weight
            v_bias = gamma * v_bias
            for x, y in zip(X, Y):
                dweight += self.grad_weight(
                    x, y, self.weight - v_weight, self.bias - v_bias
                )
                dbias += self.grad_bias(
                    x, y, self.weight - v_weight, self.bias - v_bias
                )
            v_weight += eta * dweight
            v_bias += eta * dbias
            self.weight = self.weight - v_weight
            self.bias = self.bias - v_bias
            self.append_log()

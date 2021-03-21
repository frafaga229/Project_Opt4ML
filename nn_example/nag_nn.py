from .nnetwork import NNetwork
from sklearn.metrics import accuracy_score, log_loss
import numpy as np


class NAGNetwork(NNetwork):
    def fit(self, X, Y, epochs=1, algo="GD", display_loss=False,
            eta=1, mini_batch_size=100, eps=1e-8,
            beta=0.9, beta1=0.9, beta2=0.9, gamma=0.9):
        print("===================NAG Model Fitting===================")
        for num_epoch in range(epochs):
            m = X.shape[0]

            temp_params = {}
            for i in range(1, self.num_layers + 1):
                self.update_params["v_w" + str(i)] = gamma * \
                    self.prev_update_params["v_w" + str(i)]
                self.update_params["v_b" + str(i)] = gamma * \
                    self.prev_update_params["v_b" + str(i)]
                temp_params["W" + str(i)] = self.params[
                    "W" + str(i)] - self.update_params["v_w" + str(i)]
                temp_params["B" + str(i)] = self.params[
                    "B" + str(i)] - self.update_params["v_b" + str(i)]
            self.grad(X, Y, temp_params)
            for i in range(1, self.num_layers + 1):
                self.update_params["v_w" + str(i)] = gamma * self.update_params[
                    "v_w" + str(i)] + eta * (self.gradients["dW" + str(i)] / m)
                self.update_params["v_b" + str(i)] = gamma * self.update_params[
                    "v_b" + str(i)] + eta * (self.gradients["dB" + str(i)] / m)
                self.params["W" + str(i)] -= eta * \
                    (self.update_params["v_w" + str(i)])
                self.params["B" + str(i)] -= eta * \
                    (self.update_params["v_b" + str(i)])
            self.prev_update_params = self.update_params

            Y_pred = self.predict(X)
            self.loss[num_epoch] = log_loss(np.argmax(Y, axis=1), Y_pred)
            Y_pred_train = np.argmax(Y_pred, 1)
            self.acc[num_epoch] = accuracy_score(
                Y_pred_train, np.argmax(Y, axis=1))
            print("Epoch {}: Train loss : {}  -  Train accuracy : {}".format(num_epoch,
                                                                             round(self.loss[num_epoch], 3), round(self.acc[num_epoch], 3)))

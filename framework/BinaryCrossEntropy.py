import numpy as np

class BinaryCrossEntropy():
    # Input: Y is a (N, 1) or (N,) array of binary targets (0 or 1)
    # Input: Yhat is a (N, 1) or (N,) array of predicted probabilities (from sigmoid)
    # Output: A single float value (average BCE loss)
    def eval(self, Y, Yhat):
        eps = 1e-15
        Yhat_clipped = np.clip(Yhat, eps, 1 - eps)
        return -np.mean(Y * np.log(Yhat_clipped) + (1 - Y) * np.log(1 - Yhat_clipped))

    # Output: Gradient of the loss with respect to Yhat
    def gradient(self, Y, Yhat):
        eps = 1e-15
        Yhat_clipped = np.clip(Yhat, eps, 1 - eps)
        return (Yhat_clipped - Y) / (Yhat_clipped * (1 - Yhat_clipped)) / Y.shape[0]
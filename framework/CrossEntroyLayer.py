import numpy as np

class CrossEntropy():
    # Input : Y is an NxK matrix of target values (one-hot encoded).
    # Input : Yhat is an NxK matrix of estimated probabilities (e.g. softmax output).
    # Where N >= 1 and K is the number of classes.
    # Output : A single floating point value representing average cross-entropy loss.
    def eval(self, Y, Yhat):
        # Add small epsilon for numerical stability
        eps = 1e-15
        Yhat_clipped = np.clip(Yhat, eps, 1 - eps)
        return -np.mean(np.sum(Y * np.log(Yhat_clipped), axis=1))

    # Input : Y is an NxK matrix of target values (one-hot encoded).
    # Input : Yhat is an NxK matrix of estimated probabilities.
    # Output : An NxK matrix representing the gradient of the loss with respect to Yhat.
    def gradient(self, Y, Yhat):
        # Add small epsilon for numerical stability
        eps = 1e-15
        Yhat_clipped = np.clip(Yhat, eps, 1 - eps)
        return -Y / Yhat_clipped
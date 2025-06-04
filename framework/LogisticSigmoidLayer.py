import numpy as np
from framework.Layer import Layer

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)

        # Numerically stable sigmoid
        sigmoid = np.empty_like(dataIn)
        positive = dataIn >= 0
        negative = ~positive

        # For positive values
        sigmoid[positive] = 1 / (1 + np.exp(-dataIn[positive]))
        # For negative values
        exp_x = np.exp(dataIn[negative])
        sigmoid[negative] = exp_x / (1 + exp_x)

        self.setPrevOut(sigmoid)
        return sigmoid

    def gradient(self):
        sigmoid = self.getPrevOut()
        grad = sigmoid * (1 - sigmoid)
        return grad

    def backward(self, gradIn):
        return gradIn * self.gradient()

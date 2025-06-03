from framework.Layer import Layer
import numpy as np

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        tanh = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))
        self.setPrevOut(tanh)
        return tanh
    
    def gradient(self):
        tanh = self.getPrevOut()
        grad = 1 - np.square(tanh)
        J = np.array([np.diag(row) for row in grad])
        return J
    
    def gradient2(self):
        tanh = self.getPrevOut()
        grad = 1 - np.square(tanh)
        return grad
    
    def backward2(self, gradIn):
        return gradIn * self.gradient2()
        
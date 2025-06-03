from framework.Layer import Layer
import numpy as np

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        sigmoid = 1 / (1 + np.exp(-dataIn))
        self.setPrevOut(sigmoid)
        return sigmoid
    
    #Input: None
    #Output: An Nx(KxD) tensor
    def gradient(self):
        sigmoid = self.getPrevOut()
        grad = sigmoid * (1 - sigmoid)
        J = np.array([np.diag(row) for row in grad])
        return J
    
    def gradient2(self):
        sigmoid = self.getPrevOut()
        grad = sigmoid * (1 - sigmoid)
        return grad
    
    
    def backward2(self, gradIn):
        return gradIn * self.gradient2()
    
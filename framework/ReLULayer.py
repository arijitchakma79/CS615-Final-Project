from framework.Layer import Layer
import numpy as np

class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        relu_out = np.maximum(0, dataIn)
        self.setPrevOut(relu_out)
        return relu_out
    
    def gradient(self):
        pass
    
    def backward(self, gradient):
        dataIn = self.getPrevIn()
        relu_deriv = (dataIn > 0).astype(float)
        return gradient * relu_deriv

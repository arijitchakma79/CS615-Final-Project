from framework.Layer import Layer
import numpy as np

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        return dataIn
    
    def gradient(self):
        data = self.getPrevIn()
        J = np.eye(data.shape[1])
        return J
from framework.Layer import Layer
import numpy as np

class FlatteningLayer3D(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        # Input shape: (N, C, H, W)
        self.setPrevIn(dataIn)
        N, C, H, W = dataIn.shape
        output = dataIn.reshape(N, -1, order='F')  # Flatten each sample
        self.setPrevOut(output)
        return output

    def gradient(self):
        pass

    def backward(self, gradient):
        # gradient shape: (N, C*H*W)
        dataIn = self.getPrevIn()
        N, C, H, W = dataIn.shape
        return gradient.reshape((N, C, H, W), order='F')

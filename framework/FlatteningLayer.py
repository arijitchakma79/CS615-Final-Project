from framework.Layer import Layer
import numpy as np

class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        N, H, W = dataIn.shape
        output = np.zeros((N, H * W))

        for n in range(N):
            output[n] = dataIn[n].reshape(-1, order='F')

        self.setPrevOut(output)
        return output
    
    def gradient(self):
        pass

    def backward(self, gradient):
        dataIn = self.getPrevIn()  # Shape: (N, H, W)
        N, H, W = dataIn.shape

        grad = np.zeros((N, H, W))
        for n in range(N):
            grad[n] = gradient[n].reshape((H, W), order='F')
        
        return grad
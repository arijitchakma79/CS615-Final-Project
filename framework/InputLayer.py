from framework.Layer import Layer
import numpy as np

class InputLayer(Layer):
    #Input : dataIn, an (N by D) matrix
    #Output : None
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0, keepdims=True)
        self.stdX = np.std(dataIn, axis=0, keepdims=True, ddof=1)
        
        self.stdX[self.stdX == 0] = 1
    
    #Input : dataIn, a (1 by D) matrix
    #Output : A (1 by D) matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        
        # Z-score normalization
        z = (dataIn - self.meanX) / self.stdX
        
        self.setPrevOut(z)
        return z

    def gradient(self):
        return 1 / self.stdX
from framework.Layer import Layer
import numpy as np

class FullyConnectedLayer(Layer):
    #Input: sizeIn, the number of features of data coming in
    #Input: sizeOut, the number of features of data coming out
    #Output: None

    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        limit = np.sqrt(6 / (sizeIn + sizeOut))
        self.weights = np.random.uniform(-limit, limit, size=(sizeIn, sizeOut))
        self.biases = np.random.uniform(-1e-4, 1e-4, size=(1, sizeOut))
    
    #Input: None
    #Output: The (sizeIn by sizeOut) weight matrix
    def getWeights(self):
        return self.weights
    
    #Input: The (sizeIn by sizeOut) weight matrix.
    #Output: None
    def setWeights(self, weights):
        self.weights = weights
    
    #Input: None
    #Output: The (1 by sizeOut) bias matrix.
    def getBiases(self):
        return self.biases
    
    #Input: The (1 by sizeOut) bias matrix.
    #Output: None
    def setBiases(self, biases):
        self.biases = biases

    #Input: dataIn, a (1 by D) data matrix
    #Output: A (1 by K) data matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)

        out = np.dot(dataIn, self.weights) + self.biases

        self.setPrevOut(out)
        
        return out
    
    def gradient(self):
        x = self.getPrevIn()
        grad = self.weights.T
        tensor = np.array([grad for _ in range(x.shape[0])])
        return tensor
    
    def gradient2(self):
        x = self.getPrevIn()
        grad = self.weights.T
        return grad
    
    def backward2(self, gradIn):
        return gradIn @ self.weights.T
    
    def updateWeights(self, gradIn, eta):
        N = gradIn.shape[0]
        dJdb = np.sum(gradIn, axis=0) / N
        dJdW = (self.getPrevIn().T @ gradIn) / N
        self.weights -= eta * dJdW
        self.biases -= eta * dJdb
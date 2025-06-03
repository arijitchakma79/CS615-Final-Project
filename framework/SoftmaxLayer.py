import numpy as np
from framework.Layer import Layer

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        shifted_logits = dataIn - np.max(dataIn, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        softmax_output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.setPrevOut(softmax_output)
        return softmax_output
    
    def gradient(self):
        s = self.getPrevOut()
        N, C = s.shape
        jacobians = np.zeros((N, C, C))

        for n in range(N):
            s_n = s[n].reshape(-1, 1)
            jacobians[n] = np.diagflat(s_n) -  s_n @ s_n.T
        
        return jacobians
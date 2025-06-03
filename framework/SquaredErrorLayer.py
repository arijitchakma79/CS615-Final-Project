import numpy as np

class SquaredError():
    #Input : Y is an NxK matrix of target values.
    #Input : Yhat is an NxK matrix of estimated values.
    # Where N can be any integer >=1
    #Output : A single float point value.
    def eval(self, Y, Yhat):
        return np.mean(np.square(Y - Yhat))
    
    #Input : Y is an NxK matrix of target values.
    #Input : Yhat is an NxK matrix of estimated values.
    #Output : An NxK matrix
    def gradient(self, Y, Yhat):
        return 2 * (Yhat - Y)
import numpy as np

class LogLoss():
    
    #Input : Y is an NxK matrix of target values.
    #Input : Yhat is an NxK matrix of estimated values.
    # Where N can be any integer >=1
    #Output : A single float point value.
    def eval(self, Y, Yhat):
        Yhat = np.clip(Yhat, 1e-7 , 1 - 1e-7 )
        J = -np.mean((Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat)))
        return J
    

    def gradient(self, Y, Yhat):
        Yhat = np.clip(Yhat, 1e-7, 1 - 1e-7)
        return -(Y / Yhat - (1 - Y) / (1 - Yhat))
    
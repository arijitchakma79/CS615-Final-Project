from framework.Layer import Layer
import numpy as np

class MaxPoolLayer(Layer):
    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        N, H, W = dataIn.shape
        p = self.pool_size
        s = self.stride

        out_H = (H - p) // s + 1
        out_W = (W - p) // s + 1

        output = np.zeros((N, out_H, out_W))

        for n in range(N):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * s
                    w_start = j * s
                    region = dataIn[n, h_start:h_start + p, w_start:w_start + p]
                    max_val = np.max(region)
                    output[n, i, j] = max_val

        self.setPrevOut(output)
        return output
    
    def gradient(self):
        pass

    def backward(self, gradient):
        dataIn = self.getPrevIn()
        N, H, W = dataIn.shape
        p = self.pool_size
        s = self.stride
        out_H = (H - p) // s + 1
        out_W = (W - p) // s + 1

        grad = np.zeros_like(dataIn)

        for n in range(N):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * s
                    w_start = j * s
                    region = dataIn[n, h_start:h_start + p, w_start:w_start + p]
                    max_val = np.max(region)

                    for m in range(p):
                        for n_ in range(p):
                            if(region[m, n_] == max_val):
                                grad[n, h_start + m, w_start + n_] += gradient[n, i, j]
                                break
                        else: 
                            continue
                        break
        return grad
        
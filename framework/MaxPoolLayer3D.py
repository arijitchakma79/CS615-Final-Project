from framework.Layer import Layer
import numpy as np

class MaxPoolLayer3D(Layer):
    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        N, C, H, W = dataIn.shape
        p, s = self.pool_size, self.stride

        out_H = (H - p) // s + 1
        out_W = (W - p) // s + 1

        output = np.zeros((N, C, out_H, out_W))
        self.max_indices = np.zeros((N, C, out_H, out_W, 2), dtype=int)

        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * s
                        w_start = j * s
                        region = dataIn[n, c, h_start:h_start + p, w_start:w_start + p]
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        output[n, c, i, j] = region[max_idx]
                        self.max_indices[n, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        self.setPrevOut(output)
        return output

    def backward(self, gradient):
        dataIn = self.getPrevIn()
        N, C, H, W = dataIn.shape
        p, s = self.pool_size, self.stride
        out_H, out_W = gradient.shape[2], gradient.shape[3]

        grad_input = np.zeros_like(dataIn)

        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_idx, w_idx = self.max_indices[n, c, i, j]
                        grad_input[n, c, h_idx, w_idx] += gradient[n, c, i, j]

        return grad_input

    def gradient(self):
        pass
    

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

        shape = (N, C, out_H, out_W, p, p)
        strides = (
            dataIn.strides[0],
            dataIn.strides[1],
            dataIn.strides[2] * s,
            dataIn.strides[3] * s,
            dataIn.strides[2],
            dataIn.strides[3]
        )

        self.windows = np.lib.stride_tricks.as_strided(dataIn, shape=shape, strides=strides)
        self.flat_windows = self.windows.reshape(N, C, out_H, out_W, -1)
        self.argmax = np.argmax(self.flat_windows, axis=-1)
        output = np.max(self.flat_windows, axis=-1)

        self.setPrevOut(output)
        return output

    def backward(self, gradient):
        dataIn = self.getPrevIn()
        N, C, H, W = dataIn.shape
        p, s = self.pool_size, self.stride
        out_H, out_W = gradient.shape[2], gradient.shape[3]

        grad_input = np.zeros_like(dataIn)

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * s
                w_start = j * s
                for n in range(N):
                    for c in range(C):
                        idx = self.argmax[n, c, i, j]
                        h_off = idx // p
                        w_off = idx % p
                        grad_input[n, c, h_start + h_off, w_start + w_off] += gradient[n, c, i, j]

        return grad_input

    def gradient(self):
        pass
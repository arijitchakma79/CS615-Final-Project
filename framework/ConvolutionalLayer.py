from framework.Layer import Layer
import numpy as np

class ConvolutionalLayer(Layer):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel = np.random.uniform(-1e-4, 1e-4, size=(kernel_size, kernel_size))

    def setKernels(self, kernel_matrix):
        self.kernel = kernel_matrix
    
    def getKernels(self):
        return self.kernel
    
    @staticmethod
    def crossCorrelate2D(kernel, matrix):
        H, W = matrix.shape
        kH, kW = kernel.shape
        out_H = H - kH + 1
        out_W = W - kW + 1

        output = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                region = matrix[i:i+kH, j:j+kW]
                output[i, j] = np.sum(region * kernel)

        return output
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        N, H, W = dataIn.shape
        kH, kW = self.kernel.shape
        out_H = H - kH + 1
        out_W = W - kW + 1

        output = np.zeros((N, out_H, out_W))
        for n in range(N):
            output[n] = self.crossCorrelate2D(self.kernel, dataIn[n])

        self.setPrevOut(output)
        return output
    
    def gradient(self):
        pass
    
    def backward(self, gradient):
        dataIn = self.getPrevIn()
        N, H, W = dataIn.shape
        kH, kW = self.kernel.shape
        out_H, out_W = gradient.shape[1:]

        grad_input = np.zeros_like(dataIn)

        for n in range(N):
            flipped_kernel = np.flip(np.flip(self.kernel, axis=0), axis=1)

            padded = np.pad(gradient[n], ((kH - 1, kH - 1), (kW - 1, kW - 1)))
            grad_input[n] = self.crossCorrelate2D(flipped_kernel, padded)

        return grad_input

    def updateKernels(self, gradient, learning_rate):
        dataIn = self.getPrevIn()
        N, H, W = dataIn.shape
        for n in range(N):
            dJdK = self.crossCorrelate2D(gradient[n], dataIn[n])
            self.kernel = self.kernel - learning_rate * dJdK
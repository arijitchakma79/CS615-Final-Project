from framework.Layer import Layer
import numpy as np

class ConvolutionalLayer3D(Layer):
    def __init__(self, kernel_size, in_channels, out_channels, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        limit = np.sqrt(6 / (in_channels + out_channels))
        self.kernels = np.random.uniform(-limit, limit, 
                                         size=(out_channels, in_channels, kernel_size, kernel_size))

        self.bias = np.zeros(out_channels)

        self.m = np.zeros_like(self.kernels)
        self.v = np.zeros_like(self.kernels)
        self.t = 0
        self.dJdB = 0
        self.dJdK = 0

    @staticmethod
    def crossCorrelate3D(kernel, matrix):
        C, H, W = matrix.shape
        _, kH, kW = kernel.shape
        out_H = H - kH + 1
        out_W = W - kW + 1
        out = np.zeros((out_H, out_W))

        for c in range(C):
            out += ConvolutionalLayer3D.crossCorrelate2D(kernel[c], matrix[c])

        return out

    @staticmethod
    def crossCorrelate2D(kernel, matrix):
        H, W = matrix.shape
        kH, kW = kernel.shape
        out_H = H - kH + 1
        out_W = W - kW + 1
        out = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                region = matrix[i:i+kH, j:j+kW]
                out[i, j] = np.sum(region * kernel)

        return out

    def forward(self, dataIn):
        if self.padding > 0:
            dataIn_padded = np.pad(dataIn, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            dataIn_padded = dataIn

        self.setPrevIn(dataIn)
        N, C, H, W = dataIn_padded.shape
        kH, kW = self.kernel_size, self.kernel_size
        out_H = H - kH + 1
        out_W = W - kW + 1
        output = np.zeros((N, self.out_channels, out_H, out_W))

        for n in range(N):
            for oc in range(self.out_channels):
                output[n, oc] = self.crossCorrelate3D(self.kernels[oc], dataIn_padded[n]) + self.bias[oc]

        self.setPrevOut(output)
        return output
    
    def gradient(self):
        pass

    def backward(self, gradient):
        dataIn = self.getPrevIn()
        N, C_in, H_orig, W_orig = dataIn.shape
        pad = self.padding
        kH, kW = self.kernel_size, self.kernel_size

        if pad > 0:
            dataIn_padded = np.pad(dataIn, ((0,0), (0,0), (pad, pad), (pad, pad)))
        else:
            dataIn_padded = dataIn

        N, C_in, H, W = dataIn_padded.shape
        grad_input_padded = np.zeros_like(dataIn_padded)

        self.dJdK = np.zeros_like(self.kernels)
        self.dJdB = np.zeros_like(self.bias)

        for n in range(N):
            for oc in range(self.out_channels):
                grad_out = gradient[n, oc]
                flipped_kernel = np.flip(np.flip(self.kernels[oc], axis=1), axis=2)

                for ic in range(C_in):
                    padded_grad_out = np.pad(grad_out, ((kH - 1, kH - 1), (kW - 1, kW - 1)))
                    grad_input_padded[n, ic] += self.crossCorrelate2D(flipped_kernel[ic], padded_grad_out)

                for ic in range(C_in):
                    self.dJdK[oc, ic] += self.crossCorrelate2D(grad_out, dataIn_padded[n, ic])

                self.dJdB[oc] += np.sum(grad_out)

        if pad > 0:
            grad_input = grad_input_padded[:, :, pad:-pad, pad:-pad]
        else:
            grad_input = grad_input_padded

        return grad_input

    def updateKernels(self, learning_rate):
        self.kernels -= learning_rate * self.dJdK
        self.bias -= learning_rate * self.dJdB

    def updateKernelsAdam(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * self.dJdK
        self.v = beta2 * self.v + (1 - beta2) * (self.dJdK ** 2)

        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)

        self.kernels -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        self.bias -= learning_rate * self.dJdB


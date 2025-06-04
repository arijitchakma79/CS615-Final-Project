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

    def forward(self, dataIn):
        if self.padding > 0:
            dataIn = np.pad(dataIn, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        self.setPrevIn(dataIn)
        N, C, H, W = dataIn.shape
        kH, kW = self.kernel_size, self.kernel_size
        out_H = H - kH + 1
        out_W = W - kW + 1

        output = np.zeros((N, self.out_channels, out_H, out_W))
        self.cols = np.zeros((N, C * kH * kW, out_H * out_W))

        for n in range(N):
            col = self.im2col(dataIn[n], kH, kW)
            self.cols[n] = col
            reshaped_kernel = self.kernels.reshape(self.out_channels, -1)
            output[n] = (reshaped_kernel @ col).reshape(self.out_channels, out_H, out_W) + self.bias[:, None, None]

        self.setPrevOut(output)
        return output

    def backward(self, gradient):
        dataIn = self.getPrevIn()
        N, C, H, W = dataIn.shape
        kH, kW = self.kernel_size, self.kernel_size
        out_H, out_W = gradient.shape[2], gradient.shape[3]

        self.dJdK = np.zeros_like(self.kernels)
        self.dJdB = np.zeros_like(self.bias)
        grad_input = np.zeros_like(dataIn)

        for n in range(N):
            dout_flat = gradient[n].reshape(self.out_channels, -1)
            self.dJdK += (dout_flat @ self.cols[n].T).reshape(self.kernels.shape)
            self.dJdB += np.sum(dout_flat, axis=1)

            reshaped_kernel = self.kernels.reshape(self.out_channels, -1)
            dcol = reshaped_kernel.T @ dout_flat
            grad_input[n] = self.col2im(dcol, (C, H, W), kH, kW)

        if self.padding > 0:
            return grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return grad_input

    def im2col(self, image, kH, kW):
        C, H, W = image.shape
        out_H = H - kH + 1
        out_W = W - kW + 1
        col = np.zeros((C * kH * kW, out_H * out_W))

        for y in range(out_H):
            for x in range(out_W):
                patch = image[:, y:y + kH, x:x + kW]
                col[:, y * out_W + x] = patch.flatten()

        return col

    def col2im(self, col, shape, kH, kW):
        C, H, W = shape
        out_H = H - kH + 1
        out_W = W - kW + 1
        image = np.zeros(shape)

        for y in range(out_H):
            for x in range(out_W):
                image[:, y:y + kH, x:x + kW] += col[:, y * out_W + x].reshape(C, kH, kW)

        return image

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

    def gradient(self):
        pass
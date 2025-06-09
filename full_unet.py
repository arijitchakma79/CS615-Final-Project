import numpy as np
from framework import (
    ConvolutionalLayer3D, 
    ReLULayer, 
    MaxPoolLayer3D)


def center_crop(to_crop, target):
    _, _, H, W = to_crop.shape
    _, _, Ht, Wt = target.shape
    start_H = (H - Ht) // 2
    start_W = (W - Wt) // 2
    return to_crop[:, :, start_H:start_H + Ht, start_W:start_W + Wt]


def upsample_nearest(x, scale=2):
    return x[:, :, :, :, np.newaxis].repeat(scale, axis=4) \
            .reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * scale) \
            [:, :, :, np.newaxis, :].repeat(scale, axis=3) \
            .reshape(x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)


class FullUNet:
    def __init__(self, in_channels=1):
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = MaxPoolLayer3D(pool_size=2, stride=2)

        self.enc2 = self.conv_block(64, 128)
        self.pool2 = MaxPoolLayer3D(pool_size=2, stride=2)

        self.enc3 = self.conv_block(128, 256)
        self.pool3 = MaxPoolLayer3D(pool_size=2, stride=2)

        self.enc4 = self.conv_block(256, 512)
        self.pool4 = MaxPoolLayer3D(pool_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        # Final output
        self.final_conv = ConvolutionalLayer3D(kernel_size=1, in_channels=64, out_channels=1, padding=0)

    def conv_block(self, in_channels, out_channels):
        return [
            ConvolutionalLayer3D(3, in_channels, out_channels, padding=1), ReLULayer(),
            ConvolutionalLayer3D(3, out_channels, out_channels, padding=1), ReLULayer()
        ]

    def forward_block(self, x, block):
        for layer in block:
            x = layer.forward(x)
        return x

    def backward_block(self, grad, block):
        for layer in reversed(block):
            grad = layer.backward(grad)
        return grad

    def forward(self, x):
        self.x1 = self.forward_block(x, self.enc1)
        self.x2 = self.pool1.forward(self.x1)

        self.x3 = self.forward_block(self.x2, self.enc2)
        self.x4 = self.pool2.forward(self.x3)

        self.x5 = self.forward_block(self.x4, self.enc3)
        self.x6 = self.pool3.forward(self.x5)

        self.x7 = self.forward_block(self.x6, self.enc4)
        self.x8 = self.pool4.forward(self.x7)

        self.x9 = self.forward_block(self.x8, self.bottleneck)

        self.x10_up = upsample_nearest(self.x9)
        self.x7_crop = center_crop(self.x7, self.x10_up)
        self.x10 = self.forward_block(np.concatenate([self.x10_up, self.x7_crop], axis=1), self.dec4)

        self.x11_up = upsample_nearest(self.x10)
        self.x5_crop = center_crop(self.x5, self.x11_up)
        self.x11 = self.forward_block(np.concatenate([self.x11_up, self.x5_crop], axis=1), self.dec3)

        self.x12_up = upsample_nearest(self.x11)
        self.x3_crop = center_crop(self.x3, self.x12_up)
        self.x12 = self.forward_block(np.concatenate([self.x12_up, self.x3_crop], axis=1), self.dec2)

        self.x13_up = upsample_nearest(self.x12)
        self.x1_crop = center_crop(self.x1, self.x13_up)
        self.x13 = self.forward_block(np.concatenate([self.x13_up, self.x1_crop], axis=1), self.dec1)

        self.out = self.final_conv.forward(self.x13)
        return self.out

    def backward(self, gradOut):
        grad = self.final_conv.backward(gradOut)
        grad = self.backward_block(grad, self.dec1)
        grad = self.backward_block(grad, self.dec2)
        grad = self.backward_block(grad, self.dec3)
        grad = self.backward_block(grad, self.dec4)
        grad = self.backward_block(grad, self.bottleneck)
        grad = self.backward_block(grad, self.enc4)
        grad = self.backward_block(grad, self.enc3)
        grad = self.backward_block(grad, self.enc2)
        grad = self.backward_block(grad, self.enc1)
        return grad

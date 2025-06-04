import numpy as np

# Assume you already implemented these:

from framework import (ConvolutionalLayer3D, ReLULayer, MaxPoolLayer3D)
def center_crop(to_crop, target):
    _, _, H, W = to_crop.shape
    _, _, Ht, Wt = target.shape

    start_H = (H - Ht) // 2
    start_W = (W - Wt) // 2

    return to_crop[:, :, start_H:start_H + Ht, start_W:start_W + Wt]

class SimpleUNet:
    def __init__(self, in_channels=1):
        # Encoder
        self.conv1 = ConvolutionalLayer3D(3, in_channels=in_channels, out_channels=8, padding=1)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolLayer3D(pool_size=2, stride=2)

        self.conv2 = ConvolutionalLayer3D(3, in_channels=8, out_channels=16, padding=1)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolLayer3D(pool_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = ConvolutionalLayer3D(3, in_channels=16, out_channels=32, padding=1)
        self.bottleneck_relu = ReLULayer()

        # Decoder
        self.conv3 = ConvolutionalLayer3D(3, in_channels=32 + 16, out_channels=16, padding=1)
        self.relu3 = ReLULayer()

        self.conv4 = ConvolutionalLayer3D(3, in_channels=16 + 8, out_channels=8, padding=1)
        self.relu4 = ReLULayer()

        self.conv5 = ConvolutionalLayer3D(3, in_channels=8, out_channels=8, padding=1)
        self.relu5 = ReLULayer()

        # Final conv
        self.conv_final = ConvolutionalLayer3D(1, in_channels=8, out_channels=1, padding=0)

    def upsample_layer(self, x, scale=2):
        return x[:, :, :, :, np.newaxis].repeat(scale, axis=4) \
                .reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * scale) \
                [:, :, :, np.newaxis, :].repeat(scale, axis=3) \
                .reshape(x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)

    def forward(self, x):
        # Encoder
        self.x1 = self.relu1.forward(self.conv1.forward(x))      
        self.x2 = self.pool1.forward(self.x1)                    

        self.x3 = self.relu2.forward(self.conv2.forward(self.x2)) 
        self.x4 = self.pool2.forward(self.x3)                    

        # Bottleneck
        self.x5 = self.bottleneck_relu.forward(self.bottleneck_conv.forward(self.x4)) 
        
        # Decoder
        self.x6_up = self.upsample_layer(self.x5)
        self.x3_cropped = center_crop(self.x3, self.x6_up)
        self.x6 = np.concatenate([self.x6_up, self.x3_cropped], axis=1)
        self.x6 = self.relu3.forward(self.conv3.forward(self.x6))

        self.x7_up = self.upsample_layer(self.x6)                
        self.x1_cropped = center_crop(self.x1, self.x7_up)        
        self.x7 = np.concatenate([self.x7_up, self.x1_cropped], axis=1)
        self.x7 = self.relu4.forward(self.conv4.forward(self.x7))

        self.x8 = self.relu5.forward(self.conv5.forward(self.x7))

        self.output = self.conv_final.forward(self.x8)      
        
        return self.output

    def backward(self, dJdOut):
        dJdx8 = self.conv_final.backward(dJdOut)               
        dJdx8 = self.relu5.backward(dJdx8)
        # print(dJdx8.shape)  
        dJdx7 = self.conv5.backward(dJdx8)

        dJdx7 = self.relu4.backward(dJdx7)
        dJdx7 = self.conv4.backward(dJdx7)

        # Split gradient from skip connection
        dJdx7_up, dJdx1 = np.split(dJdx7, [16], axis=1)
        dJdx7_up_down = self.downsample_like(dJdx7_up, self.relu3.getPrevOut().shape)

        dJdx6 = self.relu3.backward(dJdx7_up_down)
        dJdx6 = self.conv3.backward(dJdx6)

        dJdx6_up, dJdx3 = np.split(dJdx6, [32], axis=1)

        dJdx5 = self.downsample_like(dJdx6_up, self.x5.shape)
        dJdx3 += self.pool2.backward(dJdx5)

        dJdx3 = self.relu2.backward(dJdx3)
        dJdx3 = self.conv2.backward(dJdx3)

        dJdx2 = self.pool1.backward(dJdx3)

        dJdx1 += dJdx2
        dJdx1 = self.relu1.backward(dJdx1)
        dJdx1 = self.conv1.backward(dJdx1)

        return dJdx1

    def downsample_like(self, grad, shape_target):
        """Average pool gradient to match target shape."""
        N, C, H_out, W_out = shape_target
        scale_H = grad.shape[2] // H_out
        scale_W = grad.shape[3] // W_out
        grad_down = grad.reshape(N, C, H_out, scale_H, W_out, scale_W)
        grad_down = grad_down.mean(axis=(3, 5))
        return grad_down
    
def test_simple_unet(model_class):
    model = model_class()

    x = np.random.randn(1, 3, 64, 64).astype(np.float32)

    # Forward pass
    output = model.forward(x)
    print(f"Forward output shape: {output.shape}")

    target = np.random.randn(*output.shape).astype(np.float32)

    # Compute gradient of MSE loss w.r.t output: dJ/dOut = 2*(output - target) / N
    dJdOut = 2 * (output - target) / np.prod(output.shape)

    # Backward pass
    dJdInput = model.backward(dJdOut)
    print(f"Backward output (gradient wrt input) shape: {dJdInput.shape}")

    # Sanity checks
    assert output.shape == (1, 3, 64, 64), "Unexpected output shape"
    assert dJdInput.shape == x.shape, "Gradient shape mismatch with input"

    print("Test passed: forward and backward executed successfully with matching shapes.")    
if __name__ == '__main__' :
    model = SimpleUNet()
    test_simple_unet(SimpleUNet)
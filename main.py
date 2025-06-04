import os
import cv2
import numpy as np
from unet import SimpleUNet
import matplotlib.pyplot as plt
from framework import (LogisticSigmoidLayer, BinaryCrossEntropy)

def load_pet_dataset(image_dir, trimap_dir, split_file, img_size=(64, 64), max_samples=None):
    with open(split_file, 'r') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        img_name = line.split(' ')[0]
        image_path = os.path.join(image_dir, img_name + '.jpg')
        mask_path = os.path.join(trimap_dir, img_name + '.png')

        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)              # (1, 64, 64)

        # Load and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask == 1).astype(np.float32)          # Foreground only
        mask = np.expand_dims(mask, axis=0)            # (1, 64, 64)

        data.append((img, mask))
        if max_samples and len(data) >= max_samples:
            break

    return data

def train_unet(model, data, epochs=5, lr=1e-3):
    print("-------------Start Training---------------")
    L1 = model
    L2 = LogisticSigmoidLayer()
    L3 = BinaryCrossEntropy()
    layers = [L1, L2, L3]
    for epoch in range(epochs):
        total_loss = 0

        # --- Forward Pass ---
        for img, mask in data:
            output = img[None, :, :, :]  # (1, 1, 64, 64)
            mask = mask[None, :, :, :]  # (1, 1, 64, 64)

            for layer in layers[:-1]:
                output = layer.forward(output)

            loss = L3.eval(mask, output)
            total_loss += loss
            # --- Backward Pass ---
            gradIn = L3.gradient(mask, output)
            for layer in reversed(layers[:-1]):
                grad = layer.backward(gradIn)
                if isinstance(layer, SimpleUNet):
                    for j in [model.conv1, model.conv2, model.bottleneck_conv,
                            model.conv3, model.conv4, model.conv5, model.conv_final]:
                        j.updateKernelsAdam(lr)
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(data):.4f}")

def test_data_loader(data):
    for i, (img, mask) in enumerate(data):
        print(f"Batch {i}")
        print(f"Image shape: {img.shape}")   # Should be (batch_size, channels, height, width)
        print(f"Mask shape: {mask.shape}")   # Should match image shape or (batch_size, 1, H, W)
        print(f"Image dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
        print(f"Mask unique values: {np.unique(mask)}")  # Should be {0, 1} or {1, 2, 3} for trimaps

        # Visualize first image and mask in the batch
        img_to_show = img.transpose(1, 2, 0) if img.shape[0] == 3 else img[0]
        mask_to_show = mask[0, 0] if mask.ndim == 4 else mask[0]

        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(img_to_show, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask_to_show, cmap='gray')
        plt.axis('off')

        plt.show()

        if i == 1:  # Just test 2 batches
            break
model = SimpleUNet()
img_dir = "./oxford-iiit-pet/images"
trimap_dir = "./oxford-iiit-pet/annotations/trimaps"
split_file = "./oxford-iiit-pet/annotations/trainval.txt"
data = load_pet_dataset(img_dir, trimap_dir, split_file)

# test_data_loader(data)
train_unet(model, data[:20], epochs=10)

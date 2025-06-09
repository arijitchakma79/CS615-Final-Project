import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from unet import SimpleUNet
from framework import LogisticSigmoidLayer, BinaryCrossEntropy


def save_model(model, path="simple_unet_weights.pkl"):
    weights = {}
    for i, layer in enumerate([model.conv1, model.conv2, model.bottleneck_conv,
                               model.conv3, model.conv4, model.conv5, model.conv_final]):
        weights[f"conv{i+1}_weights"] = layer.kernels
        weights[f"conv{i+1}_bias"] = layer.bias
    with open(path, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Model saved to {path}")


def load_model(model, path="simple_unet_weights.pkl"):
    with open(path, 'rb') as f:
        weights = pickle.load(f)
    for i, layer in enumerate([model.conv1, model.conv2, model.bottleneck_conv,
                               model.conv3, model.conv4, model.conv5, model.conv_final]):
        layer.kernels = weights[f"conv{i+1}_weights"]
        layer.bias = weights[f"conv{i+1}_bias"]
    print(f"Model loaded from {path}")


def load_pet_dataset(image_dir, trimap_dir, split_file, img_size=(64, 64), max_samples=None):
    with open(split_file, 'r') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        img_name = line.split(' ')[0]
        if img_name.startswith("._"):
            continue
        image_path = os.path.join(image_dir, img_name + '.jpg')
        mask_path = os.path.join(trimap_dir, img_name + '.png')

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (3, H, W)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask == 1).astype(np.float32)  # Binary: foreground only
        mask = np.expand_dims(mask, axis=0)   # (1, H, W)

        data.append((img, mask))
        if max_samples and len(data) >= max_samples:
            break

    return data


def split_dataset(data, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    split = int(len(data) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    return train_data, val_data


def train_unet(model, train_data, epochs=10, lr=1e-3):
    print("-------------Start Training---------------")
    L1 = model
    L2 = LogisticSigmoidLayer()
    L3 = BinaryCrossEntropy()
    layers = [L1, L2, L3]

    train_losses = []

    for epoch in range(epochs):
        total_loss = 0

        for img, mask in train_data:
            output = img[None, :, :, :]
            mask = mask[None, :, :, :]

            for layer in layers[:-1]:
                output = layer.forward(output)

            loss = L3.eval(mask, output)
            total_loss += loss

            gradIn = L3.gradient(mask, output)
            for layer in reversed(layers[:-1]):
                grad = layer.backward(gradIn)
                if isinstance(layer, SimpleUNet):
                    for j in [model.conv1, model.conv2, model.bottleneck_conv,
                              model.conv3, model.conv4, model.conv5, model.conv_final]:
                        j.updateKernelsAdam(lr)
                gradIn = grad

        avg_train_loss = total_loss / len(train_data)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")

    save_model(model)

    # Plot training loss
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_predictions(model, val_data, output_dir="val_predictions"):
    os.makedirs(output_dir, exist_ok=True)

    L1 = model
    L2 = LogisticSigmoidLayer()

    for idx, (img, mask) in enumerate(val_data):
        img_batch = img[None, :, :, :]
        output = L1.forward(img_batch)
        output = L2.forward(output)

        pred_mask = (output[0, 0] > 0.5).astype(np.uint8) * 255
        gt_mask = (mask[0] > 0.5).astype(np.uint8) * 255
        rgb_img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, f"{idx}_image.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{idx}_gt.png"), gt_mask)
        cv2.imwrite(os.path.join(output_dir, f"{idx}_pred.png"), pred_mask)

    print(f"Saved {len(val_data)} validation predictions to '{output_dir}/'")


def test_data_loader(data):
    for i, (img, mask) in enumerate(data):
        print(f"Sample {i}")
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")

        img_to_show = img.transpose(1, 2, 0)
        mask_to_show = mask[0]

        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(img_to_show)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask_to_show, cmap='gray')
        plt.axis('off')

        plt.show()

        if i == 1:
            break
def load_pet_dataset_by_class(image_dir, trimap_dir, split_file, class_prefix, img_size=(64, 64), max_samples=None):
    with open(split_file, 'r') as f:
        lines = f.read().splitlines()

    data = []
    for line in lines:
        img_name = line.split()[0]

        # ✅ Match all images of the same breed/class
        if not img_name.startswith(class_prefix):
            continue

        image_path = os.path.join(image_dir, img_name + '.jpg')
        mask_path = os.path.join(trimap_dir, img_name + '.png')

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask == 1).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        data.append((img, mask))

        if max_samples and len(data) >= max_samples:
            break

    return data



if __name__ == "__main__":
    model = SimpleUNet(in_channels=3)
    img_dir = "./oxford-iiit-pet/images"
    trimap_dir = "./oxford-iiit-pet/annotations/trimaps"
    split_file = "./oxford-iiit-pet/annotations/trainval.txt"

    # Load only the first 20 Abyssinian images
    data = load_pet_dataset_by_class(
        image_dir=img_dir,
        trimap_dir=trimap_dir,
        split_file=split_file,
        class_prefix="Abyssinian_",
        img_size=(64, 64),
        max_samples=50
    )
    print(f"Loaded {len(data)} samples for Abyssinian")

    # Train for 100 epochs
    train_unet(model, data, epochs=1000, lr=0.001)

    # Evaluate on the same data used for training
    save_predictions(model, data, output_dir="train_predictions")


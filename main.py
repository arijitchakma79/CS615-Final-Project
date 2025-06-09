import numpy as np
import matplotlib.pyplot as plt
from unet import SimpleUNet
from framework import (
    LogisticSigmoidLayer, 
    BinaryCrossEntropy
)
from utils.helper import (
    set_seed,
    load_pet_dataset,
    load_pet_dataset_by_class,
    split_dataset,
    save_predictions,
    test_data_loader
)


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


    # Plot training loss
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    set_seed(42)
    model = SimpleUNet(in_channels=3)
    img_dir = "./oxford-iiit-pet/images"
    trimap_dir = "./oxford-iiit-pet/annotations/trimaps"
    split_file = "./oxford-iiit-pet/annotations/trainval.txt"

    
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


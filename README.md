# 🐾 U-Net: Animal Recognition

Team Members:
Antonio Gallego Bernal (ag4258), Arijit Chakma (ac4393), Lam Nguyen (ltn45), Anh Minh Tran (at3654)

## 📌 Project Overview

This project implements a U-Net based semantic segmentation model from scratch to detect and segment **cats and dogs** in images using the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). Our goal is to build a robust and efficient pipeline that can accurately separate pet pixels from the background across diverse breeds, lighting conditions, and partial occlusions.

## 🎯 Objectives

* Accurately **segment cats and dogs** at the pixel level from complex backgrounds.
* Provide a foundational tool for **real-time animal tracking** in video footage.
* Learn and build a **custom U-Net architecture** from the ground up.

## 📚 Dataset

We use the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which contains:

* 37 pet breeds (cats and dogs)
* Over 7,000 images with pixel-level annotations
* Significant variability in pose, lighting, and background

## 🧠 Model Architecture

We implemented the **U-Net** architecture:

* Encoder-decoder structure with skip connections
* Captures both spatial and semantic information
* Effective for dense prediction tasks like segmentation
* Enhance the convolution layer by replacing cross-correlation with the im2con method.

Why U-Net?

* Proven success in biomedical and natural image segmentation
* Balanced model size and accuracy
* Great learning experience for implementing from scratch

## 🧪 Features

* 🛠 Built from scratch using only NumPy and custom layers
* 📈 Supports training, validation, and testing modes
* 🔁 Ready for future extensions such as real-time inference or larger animal datasets

## 🔄 Future Work

* Expand to more animal classes (not just pets)
* Introduce data augmentation for robustness (rotation, occlusion)
* Integrate real-time segmentation on video feeds
* Experiment with more advanced U-Net variants (e.g., attention U-Net, nnU-Net)

## 📖 References

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
* [nnU-Net: Self-adapting Framework for Medical Image Segmentation](https://www.nature.com/articles/s41592-020-01008-z)
* [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* [Stanford CS231n Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/#conv)

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/unet-animal-recognition.git
cd unet-animal-recognition
```
* Running the main.py for training and evaluating results
* The result will be saved to train_predictions.
## 🖼️ Sample Results

*Include sample before/after segmentation images here if available.*

---

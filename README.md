# 🐾 U-Net: Animal Foreground-Background Image Segmentation

**Team Members:**  
Antonio Gallego Bernal (ag4258)  
Arijit Chakma (ac4393)  
Lam Nguyen (ltn45)  
Anh Minh Tran (at3654)

---

## 📌 Project Overview

This project implements a **U-Net-based semantic segmentation model from scratch** to detect and segment **cats and dogs** in images using the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).  
Our goal is to build an efficient pipeline that can accurately distinguish pet pixels from the background, handling diverse breeds, lighting conditions, and partial occlusions.

---

## 🎯 Objectives

- 🎯 Accurately **segment cats and dogs** at the pixel level.
- 🐕 Provide a foundation for **real-time animal tracking**.
- 🔧 Build a **custom U-Net** using only NumPy — no PyTorch or TensorFlow.

---

## 📚 Dataset

We use the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which contains:
- 37 pet breeds (cats and dogs)
- Over 7,000 RGB images with pixel-level annotations
- High variability in size, pose, and lighting conditions

---

## 🧠 Model Architecture

Our implementation follows the **U-Net architecture**, featuring:
- An **encoder-decoder structure** with skip connections
- Spatial + semantic information preservation
- **Optimized 3D convolution** using `im2col`-style matrix multiplication
- A custom implementation of the **ADAM optimizer** for weight updates

### Why U-Net?
- Proven success in biomedical and natural image segmentation
- Balanced tradeoff between model complexity and accuracy
- Excellent hands-on learning opportunity for building deep learning tools from scratch

---

## 🧪 Features

- 🛠 Built entirely with **NumPy** and custom convolutional/pooling layers
- 📈 Training pipeline supports basic evaluation and visualization
- 🚀 Easily extendable to real-time or large-scale applications
- 🔧 Optimized layers for **faster computation**, including:
  - `FasterConvolutionalLayer3D`
  - `FasterMaxPoolLayer3D`
  - Custom `ADAM` optimization step per layer

---

## 🚀 Getting Started

### 🧩 Requirements

- Python 3.8+
- NumPy
- OpenCV (`opencv-python`)
- Matplotlib

### 🔧 Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/unet-animal-recognition.git
cd unet-animal-recognition

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate         # On Unix/macOS
# OR
.venv\Scripts\activate            # On Windows

# 3. Install required packages
pip install -r requirements.txt
### ▶️ Run Training

```bash
python main.py
```

This will:
- Load a subset of Abyssinian cat images (can be changed in `main.py`)
- Train the Full U-Net model for 100 epochs
- Save predicted segmentation masks to the `train_predictions/` directory

---

## 🖼️ Directory Structure

```
project-root/
├── framework/              # All custom layers (Conv, Pool, etc.)
├── utils.py                # Helper functions (data loading, prediction, etc.)
├── full_unet_model.py      # Full U-Net class (3-level deep encoder-decoder)
├── main.py                 # Training + prediction entry point
├── oxford-iiit-pet/
│   ├── images/             # Pet images (JPEG format)
│   └── annotations/
│       └── trimaps/        # Pixel-level masks (PNG format)
│       └── trainval.txt    # Train split list
├── train_predictions/      # Output predictions (PNG mask overlays)
├── requirements.txt        # Python dependencies
```

---

## 🔄 Future Work

- Expand to more animal types (wildlife, birds, etc.)
- Add real-time augmentation (flip, rotate, occlude, lighting)
- Integrate **live video segmentation**
- Replace simple upsampling with transposed convolution
- Experiment with:
  - Residual U-Net
  - Attention U-Net
  - nnU-Net auto-configuring pipelines

---

## 📖 References

- [U-Net: Biomedical Image Segmentation (Ronneberger et al.)](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Stanford CS231n Notes on CNNs](https://cs231n.github.io/convolutional-networks/)
- [nnU-Net: A self-configuring method for segmentation](https://www.nature.com/articles/s41592-020-01008-z)

---

## 🖼️ Sample Results

Include sample segmentation result images in the `train_predictions/` folder such as:

- Input image
- Ground truth mask
- Predicted mask (binary or overlayed)

You can use `matplotlib` or `OpenCV` to visualize and compare.

---

Happy Segmenting! 🐶🐱


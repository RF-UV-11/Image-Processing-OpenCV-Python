# 🖼️ Image Processing with OpenCV

This project demonstrates various image processing techniques using OpenCV, organized in a modular structure. The classes are designed to handle different aspects of image processing, enhancement, geometric transformations, segmentation, feature detection, and object detection.

## 📜 Table of Contents
- [Introduction](#introduction)
- [Classes and Methods](#classes-and-methods)
  - [ImageProcessor](#imageprocessor)
  - [ImageEnhancer](#imageenhancer)
  - [GeometricTransformer](#geometrictransformer)
  - [ImageSegmenter](#imagesegmenter)
  - [FeatureDetector](#featuredetector)
  - [FaceDetector](#facedetector)
- [Usage](#usage)
  - [Example](#example)
- [Use Cases](#use-cases)
- [Requirements](#requirements)
- [Installation](#installation)
- [Scripts](#scripts)

## 📖 Introduction

This project is a comprehensive guide to image processing using `OpenCV`. Each module is divided into classes that handle specific tasks. The project covers:

1. Basic image processing operations
2. Image enhancement techniques
3. Geometric transformations
4. Image segmentation
5. Feature detection
6. Face detection using Haar cascades

## 🛠️ Classes and Methods

### ImageProcessor

The `ImageProcessor` class handles basic image processing operations such as loading, displaying, resizing, cropping, and rotating images.

#### Methods:

- **`__init__(self, image_path)`**: Initializes the class with the image path.
- **`display_image(self, image, title='Image')`**: Displays the image using `matplotlib`.
- **`resize_image(self, size=(300, 300))`**: Resizes the image to the given size.
- **`crop_image(self, start_row=50, end_row=200, start_col=50, end_col=200)`**: Crops the image to the specified region.
- **`rotate_image(self, angle=45)`**: Rotates the image by the specified angle.

#### Use Cases:

- **Resizing**: Adjusting the size of an image to fit specific dimensions for applications like thumbnails, icons, or input to neural networks.
- **Cropping**: Extracting a specific region of interest from an image, such as a face in a portrait.
- **Rotating**: Correcting the orientation of an image or creating artistic effects.

### ImageEnhancer

The `ImageEnhancer` class is designed to enhance images using various techniques like histogram equalization, blurring, and sharpening.

#### Methods:

- **`__init__(self, image)`**: Initializes the class with the image.
- **`histogram_equalization(self)`**: Applies histogram equalization to the grayscale image.
- **`blur_image(self, kernel_size=(7, 7))`**: Applies Gaussian blur to the image.
- **`sharpen_image(self)`**: Sharpens the image using a kernel.

#### Use Cases:

- **Histogram Equalization**: Enhancing the contrast of an image, useful in medical imaging and satellite imagery.
- **Blurring**: Reducing noise or creating a smooth effect in images, often used in preprocessing steps for machine learning models.
- **Sharpening**: Enhancing the edges in an image, useful in applications like OCR (Optical Character Recognition).

### GeometricTransformer

The `GeometricTransformer` class applies geometric transformations like translation, scaling, affine, and perspective transformations.

#### Methods:

- **`__init__(self, image)`**: Initializes the class with the image.
- **`translate_image(self, x=50, y=50)`**: Translates the image by the given x and y offsets.
- **`scale_image(self, fx=0.5, fy=0.5)`**: Scales the image by the given factors.
- **`affine_transform(self, pts1, pts2)`**: Applies an affine transformation to the image.
- **`perspective_transform(self, pts1, pts2)`**: Applies a perspective transformation to the image.

#### Use Cases:

- **Translation**: Shifting the position of an image, useful in data augmentation for machine learning.
- **Scaling**: Resizing an image while maintaining its aspect ratio, important in graphics and image processing.
- **Affine and Perspective Transformation**: Correcting distortions or applying specific transformations to images for tasks like document scanning.

### ImageSegmenter

The `ImageSegmenter` class segments images using techniques like thresholding, adaptive thresholding, contour detection, and the watershed algorithm.

#### Methods:

- **`__init__(self, image)`**: Initializes the class with the image.
- **`thresholding(self, threshold=127)`**: Applies simple thresholding to the grayscale image.
- **`adaptive_thresholding(self)`**: Applies adaptive Gaussian thresholding to the grayscale image.
- **`contour_detection(self, threshold_image)`**: Detects contours in the thresholded image.
- **`watershed_algorithm(self)`**: Applies the watershed algorithm for image segmentation.

#### Use Cases:

- **Thresholding**: Binarizing an image for applications like barcode detection and OCR.
- **Adaptive Thresholding**: Handling varying lighting conditions in images, useful in document processing.
- **Contour Detection**: Finding and drawing contours in images, used in object detection and shape analysis.
- **Watershed Algorithm**: Segmenting overlapping objects, important in medical image analysis.

### FeatureDetector

The `FeatureDetector` class detects features in images using the ORB (Oriented FAST and Rotated BRIEF) detector.

#### Methods:

- **`__init__(self, image)`**: Initializes the class with the image.
- **`detect_orb_features(self)`**: Detects ORB features in the grayscale image.

#### Use Cases:

- **ORB Features**: Detecting and matching keypoints in images for applications like image stitching and 3D reconstruction.

### FaceDetector

The `FaceDetector` class detects objects in images using Haar cascades.

#### Methods:

- **`__init__(self, cascade_path='haarcascade_frontalface_default.xml')`**: Initializes the class with the Haar cascade path.
- **`detect_faces(self, image)`**: Detects faces in the given image.

#### Use Cases:

- **Face Detection**: Detecting faces in images and videos, used in security systems and photo organization.

## 🛠️ Requirements

- Python 3.x
- OpenCV
- Matplotlib
- NumPy

## 💻 Installation

Install the required packages:

```bash
pip install opencv-python matplotlib numpy
```

## 🚀 Usage

1. Clone the repository.
2. Place your images in the project directory.
3. Update the image paths in `main.py`.
4. Run `main.py` to see the results.

### Example

To run the program:

```bash
python main.py
```

## 📊 Use Cases

| Technique                | Use Cases                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------|
| Resizing                 | Adjusting image sizes for thumbnails, icons, or neural network inputs.                         |
| Cropping                 | Extracting regions of interest, like faces in portraits.                                       |
| Rotating                 | Correcting image orientation or creating artistic effects.                                     |
| Histogram Equalization   | Enhancing contrast in medical imaging and satellite imagery.                                   |
| Blurring                 | Reducing noise or smoothing images, often for preprocessing in machine learning models.        |
| Sharpening               | Enhancing edges in images, useful in OCR applications.                                         |
| Translation              | Shifting image positions, useful in data augmentation.                                         |
| Scaling                  | Resizing images while maintaining aspect ratios.                                               |
| Affine/Perspective Transform | Correcting distortions or applying transformations for tasks like document scanning.         |
| Thresholding             | Binarizing images for barcode detection and OCR.                                               |
| Adaptive Thresholding    | Handling varying lighting conditions in document processing.                                   |
| Contour Detection        | Finding and drawing contours for object detection and shape analysis.                         |
| Watershed Algorithm      | Segmenting overlapping objects, important in medical image analysis.                          |
| ORB Features             | Detecting and matching keypoints for image stitching and 3D reconstruction.                    |
| Face Detection           | Detecting faces in images and videos, used in security systems and photo organization.         |

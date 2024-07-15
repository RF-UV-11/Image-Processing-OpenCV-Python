import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ImageEnhancer:
    """Class for enhancing images using various techniques."""

    def __init__(self, image):
        """Initialize the ImageEnhancer with the given image."""
        self.image = image
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def histogram_equalization(self):
        """Apply histogram equalization to the grayscale image."""
        return cv2.equalizeHist(self.gray_image)

    def plot_histograms(self, equalized_image):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        sns.histplot(np.ravel(gray_image), bins=256, kde=False, color='blue', ax=axs[0])
        axs[0].set_title('Original Histogram')
        axs[0].set_xlabel('Pixel Intensity')
        axs[0].set_ylabel('Frequency')
        
        sns.histplot(np.ravel(equalized_image), bins=256, kde=False, color='blue', ax=axs[1])
        axs[1].set_title('Equalized Histogram')
        axs[1].set_xlabel('Pixel Intensity')
        axs[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

    def blur_image(self, kernel_size=(7, 7)):
        """Apply Gaussian blur to the image."""
        return cv2.GaussianBlur(self.image, kernel_size, 0)

    def sharpen_image(self):
        """Sharpen the image using a kernel."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(self.image, -1, kernel)

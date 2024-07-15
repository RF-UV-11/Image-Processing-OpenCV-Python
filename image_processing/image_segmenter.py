import cv2
import numpy as np

class ImageSegmenter:
    """Class for segmenting images using various techniques."""

    def __init__(self, image):
        """Initialize the ImageSegmenter with the given image."""
        self.image = image
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def thresholding(self, threshold=127):
        """Apply simple thresholding to the grayscale image."""
        _, thresh_image = cv2.threshold(self.gray_image, threshold, 255, cv2.THRESH_BINARY)
        return thresh_image

    def adaptive_thresholding(self):
        """Apply adaptive Gaussian thresholding to the grayscale image."""
        return cv2.adaptiveThreshold(self.gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def contour_detection(self, threshold_image):
        """Detect contours in the thresholded image."""
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = self.image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
        return contour_image

    def watershed_algorithm(self):
        """Apply the watershed algorithm for image segmentation."""
        markers = np.zeros_like(self.gray_image, dtype=np.int32)
        markers[self.gray_image < 128] = 1
        markers[self.gray_image > 200] = 2
        cv2.watershed(self.image, markers)
        watershed_image = self.image.copy()
        watershed_image[markers == -1] = [0, 0, 255]
        return watershed_image

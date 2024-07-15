import cv2
import numpy as np

class GeometricTransformer:
    """Class for applying geometric transformations to images."""

    def __init__(self, image):
        """Initialize the GeometricTransformer with the given image."""
        self.image = image
        (self.h, self.w) = image.shape[:2]

    def translate_image(self, x=50, y=50):
        """Translate the image by the given x and y offsets."""
        M = np.float32([[1, 0, x], [0, 1, y]])
        return cv2.warpAffine(self.image, M, (self.w, self.h))

    def scale_image(self, fx=0.5, fy=0.5):
        """Scale the image by the given factors."""
        return cv2.resize(self.image, None, fx=fx, fy=fy)

    def affine_transform(self, pts1, pts2):
        """Apply an affine transformation to the image."""
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(self.image, M, (self.w, self.h))

    def perspective_transform(self, pts1, pts2):
        """Apply a perspective transformation to the image."""
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(self.image, M, (300, 300))

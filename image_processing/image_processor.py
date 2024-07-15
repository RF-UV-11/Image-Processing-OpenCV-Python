import cv2
import matplotlib.pyplot as plt

class ImageProcessor:
    """Class for basic image processing operations using OpenCV."""

    def __init__(self, image_path):
        """Initialize the ImageProcessor with the given image path."""
        self.image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def display_image(self, image, title='Image'):
        """Display an image using matplotlib."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def resize_image(self, size=(300, 300)):
        """Resize the image to the given size."""
        return cv2.resize(self.image, size)

    def crop_image(self, start_row=50, end_row=200, start_col=50, end_col=200):
        """Crop the image to the specified region."""
        return self.image[start_row:end_row, start_col:end_col]

    def rotate_image(self, angle=45):
        """Rotate the image by the specified angle."""
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(self.image, M, (w, h))

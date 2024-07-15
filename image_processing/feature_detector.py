import cv2

class FeatureDetector:
    """Class for detecting features in images."""

    def __init__(self, image):
        """Initialize the FeatureDetector with the given image."""
        self.image = image
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def detect_orb_features(self):
        """Detect ORB features in the grayscale image."""
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(self.gray_image, None)
        keypoint_image = cv2.drawKeypoints(self.image, keypoints, None, color=(0, 255, 0))
        return keypoint_image

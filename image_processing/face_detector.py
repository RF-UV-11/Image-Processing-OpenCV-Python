import cv2

class FaceDetector:
    """Class for detecting objects in images using Haar cascades."""

    def __init__(self, cascade_path='haarcascade_frontalface_default.xml'):
        """Initialize the ObjectDetector with the Haar cascade path."""
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image):
        """Detect faces in the given image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image

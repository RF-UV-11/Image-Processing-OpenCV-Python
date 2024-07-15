import cv2
import numpy as np
from image_processor import ImageProcessor
from image_enhancer import ImageEnhancer
from geometric_transformer import GeometricTransformer
from image_segmenter import ImageSegmenter
from feature_detector import FeatureDetector
from face_detector import FaceDetector

def main():
    # Initialize the ImageProcessor with an example image
    image_path = 'image.png'
    processor = ImageProcessor(image_path)

    # Display the original image
    processor.display_image(processor.image, 'Original Image')

    # Perform image resizing, cropping, and rotation
    resized_image = processor.resize_image()
    cropped_image = processor.crop_image()
    rotated_image = processor.rotate_image()

    # Display processed images
    processor.display_image(resized_image, 'Resized Image')
    processor.display_image(cropped_image, 'Cropped Image')
    processor.display_image(rotated_image, 'Rotated Image')

    # Initialize the ImageEnhancer with the original image
    enhancer = ImageEnhancer(processor.image)

    # Perform histogram equalization, blurring, and sharpening
    equalized_image = enhancer.histogram_equalization()
    blurred_image = enhancer.blur_image()
    sharpened_image = enhancer.sharpen_image()

    # Display enhanced images
    processor.display_image(enhancer.gray_image, 'Gray Image')
    processor.display_image(equalized_image, 'Equalized Image')
    processor.display_image(blurred_image, 'Blurred Image')
    processor.display_image(sharpened_image, 'Sharpened Image')

    # Initialize the GeometricTransformer with the original image
    transformer = GeometricTransformer(processor.image)

    # Perform translation, scaling, affine, and perspective transformations
    translated_image = transformer.translate_image()
    scaled_image = transformer.scale_image()
    affine_image = transformer.affine_transform(
        np.float32([[50, 50], [200, 50], [50, 200]]),
        np.float32([[10, 100], [200, 50], [100, 250]])
    )
    perspective_image = transformer.perspective_transform(
        np.float32([[56, 65], [368, 52], [28, 387], [389, 390]]),
        np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    )

    # Display transformed images
    processor.display_image(translated_image, 'Translated Image')
    processor.display_image(scaled_image, 'Scaled Image')
    processor.display_image(affine_image, 'Affine Transformed Image')
    processor.display_image(perspective_image, 'Perspective Transformed Image')

    # Initialize the ImageSegmenter with the original image
    segmenter = ImageSegmenter(processor.image)

    # Perform thresholding, adaptive thresholding, contour detection, and watershed
    thresh_image = segmenter.thresholding()
    adaptive_thresh_image = segmenter.adaptive_thresholding()
    contour_image = segmenter.contour_detection(thresh_image)
    watershed_image = segmenter.watershed_algorithm()

    # Display segmented images
    processor.display_image(thresh_image, 'Threshold Image')
    processor.display_image(adaptive_thresh_image, 'Adaptive Threshold Image')
    processor.display_image(contour_image, 'Contour Image')
    processor.display_image(watershed_image, 'Watershed Image')

    # Initialize the FeatureDetector with the original image
    feature_detector = FeatureDetector(processor.image)

    # Detect ORB features
    keypoint_image = feature_detector.detect_orb_features()

    # Display keypoints
    processor.display_image(keypoint_image, 'Keypoints Image')

    # Initialize the ObjectDetector with the Haar cascade for face detection
    face_detector = FaceDetector()

    # Load a test image for face detection
    test_image_path = 'test_face_image.png'
    test_image = cv2.imread(test_image_path)

    # Perform face detection
    detected_faces_image = face_detector.detect_faces(test_image)

    # Display the image with detected faces
    processor.display_image(detected_faces_image, 'Detected Faces')

if __name__ == "__main__":
    main()

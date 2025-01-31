# pixelx/visionx_lib/image/edge_detection.py

from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import cv2, np, ImageType


def apply_canny_edge_detection(
        image: ImageType, threshold_lower: int | None = None,
        threshold_higher: int | None = None,
        sigma: float | None = 0.3) -> ImageType:
    """Perform Canny edge detection on the input image."""
    validations.validate_threshold(threshold_lower, threshold_higher)

    if not threshold_lower and not threshold_higher:
        # Compute median of the pixel intensities
        median = np.median(image)
        # Adjust thresholds based on median
        threshold_lower = max(0, int((1.0 - sigma) * median))
        threshold_higher = max(255, int((1.0 + sigma) * median))

    return cv2.Canny(image, threshold1=threshold_lower,
                     threshold2=threshold_higher)


def apply_detect_hough_lines(image: ImageType, rho: float, theta_degrees: float,
                             threshold: int, min_line_length: float,
                             max_line_gap: float) -> ImageType:
    theta = np.pi / theta_degrees
    return cv2.HoughLinesP(
        image, rho, theta, threshold, np.array([]),
        minLineLength=min_line_length, maxLineGap=max_line_gap)

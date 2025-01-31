# pixelx/visionx_lib/image/threshold.py

from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import cv2, np, ImageType
from pixelx.visionx_lib.image.color_conversion import ensure_grayscale


def apply_threshold(
        image: ImageType, thresh: int = 127, maxval: int = 255) -> ImageType:
    """Apply binary thresholding to the input image."""

    validations.validate_threshold_values(thresh, maxval)
    image = ensure_grayscale(image)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)

    return binary_image


def apply_adaptive_threshold(img: ImageType, max_value: int = 255,
                             block_size: int = 11, c: int = 2) -> np.ndarray:
    """Apply adaptive thresholding to the image."""
    validations.validate_adaptive_threshold_params(max_value, block_size, c)
    img = ensure_grayscale(img)

    return cv2.adaptiveThreshold(
        img, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c)

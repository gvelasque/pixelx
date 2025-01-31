# pixelx/visionx_lib/image/color_conversion.py

from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import cv2, ImageType


def convert_to_rgb2grayscale(image: ImageType) -> ImageType:
    """Convert an RGB image to grayscale."""
    validations.validate_bgr_channels(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_bgr2rgb(image: ImageType) -> ImageType:
    """Convert a BGR image to RGB format."""
    validations.validate_bgr_channels(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_grayscale(image: ImageType) -> ImageType:
    """Convert an image to grayscale if it's not already."""
    if image.ndim == 3 and image.shape[2] == 3:
        return convert_to_rgb2grayscale(image)
    elif image.ndim == 3 and image.shape[2] == 1:
        # Single-channel stored in 3D
        return image.reshape(image.shape[0], image.shape[1])

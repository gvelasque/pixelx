# pixelx/visionx_lib/image/filters.py

from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import cv2, np, ImageType


def apply_gaussian_blur(image: ImageType,
                        kernel_size: list[int, int] = (5, 5),
                        deviation: float = 0) -> ImageType:
    """Apply Gaussian blur to the input image."""
    validations.validate_kernel_size(kernel_size)
    return cv2.GaussianBlur(image, ksize=kernel_size, sigmaX=deviation)


def apply_blur(image: ImageType, kernel_size: list[int, int]) -> ImageType:
    """Apply averaging blur to the input image."""
    validations.validate_kernel_size(kernel_size)
    return cv2.blur(image, kernel_size)


def apply_median_blur(image: ImageType, kernel_size: int) -> ImageType:
    """Apply median blur to the input image."""
    validations.validate_kernel_size(kernel_size)
    return cv2.medianBlur(image, kernel_size)


def apply_bilateral_filter(img: ImageType, diameter: int, sigma_color: float,
                           sigma_space: float) -> ImageType:
    """Apply bilateral filtering to the input image."""
    validations.validate_bilateral_filter_params(diameter, sigma_color,
                                                 sigma_space)
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)


def filter_2d(img: ImageType, kernel: np.ndarray) -> ImageType:
    """Apply a 2D convolution filter to the image."""
    validations.validate_2d_filter_kernel(kernel)
    return cv2.filter2D(img, -1, kernel)

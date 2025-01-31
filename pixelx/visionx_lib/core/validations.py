# pixelx/visionx_lib/core/validations.py

from pixelx.visionx_lib.core.base import np, ImageType
from pixelx.visionx_lib.core.enums import RotateType, FlipType, ChannelType


# ----------------- General Image Validation -----------------

def validate_image(img: ImageType, img_path: str) -> None:
    """Validate if the input is a proper OpenCV image."""
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    if not isinstance(img, np.ndarray):
        raise TypeError("Input image must be a NumPy ndarray.")

    if img.ndim not in [2, 3]:
        raise ValueError(
            "Invalid image dimensions. Must be 2D (grayscale) or 3D (color).")

    if img.size == 0:
        raise ValueError("Input image is empty.")

    if img.dtype != np.uint8:
        raise TypeError("Image must be of type np.uint8.")


# ----------------- Dimension & Coordinate Validation -----------------

def validate_dimensions(width: int | float, height: int | float) -> None:
    """Validate width and height values."""
    if not isinstance(width, (int, float)) or width <= 0:
        raise ValueError("Width must be a positive integer.")

    if not isinstance(height, (int, float)) or height <= 0:
        raise ValueError("Height must be a positive integer.")


def validate_coordinates(image: ImageType, y_start: int, y_end: int,
                         x_start: int, x_end: int) -> None:
    """Validate that the given coordinates are within valid bounds for a 2D operation."""
    from .utils import get_image_dimensions

    if not all(isinstance(i, int) for i in [y_start, y_end, x_start, x_end]):
        raise ValueError("Coordinates must be integers.")

    height, width = get_image_dimensions(image)

    if not (0 <= y_start < y_end <= height and 0 <= x_start < x_end <= width):
        raise ValueError(
            "Invalid coordinates. Ensure 0 ≤ start < end and within bounds.")


def validate_roi_inputs(start_point=None, end_point=None, center=None,
                        radius=None) -> None:
    """General validation for ROI mask inputs."""
    if start_point is not None and (
            not isinstance(start_point, tuple) or len(start_point) != 2):
        raise ValueError("start_point must be a tuple (x, y).")

    if end_point is not None and (
            not isinstance(end_point, tuple) or len(end_point) != 2):
        raise ValueError("end_point must be a tuple (x, y).")

    if center is not None and (
            not isinstance(center, tuple) or len(center) != 2):
        raise ValueError("center must be a tuple (x, y).")

    if radius is not None and (not isinstance(radius, int) or radius <= 0):
        raise ValueError("radius must be a positive integer.")


def validate_transformation_matrix(matrix: np.ndarray,
                                   expected_shape: tuple[int, int]) -> None:
    """Validate the transformation matrix."""
    if not isinstance(matrix, np.ndarray) or matrix.shape != expected_shape:
        raise ValueError(
            f"Transformation matrix must be a NumPy array of shape {expected_shape}.")


def validate_perspective_points(src_points: np.ndarray,
                                dst_points: np.ndarray) -> None:
    """Validate source and destination points for perspective transformation."""
    if not isinstance(src_points, np.ndarray) or not isinstance(dst_points,
                                                                np.ndarray):
        raise TypeError("src_points and dst_points must be NumPy arrays.")
    if src_points.shape != (4, 2) or dst_points.shape != (4, 2):
        raise ValueError("src_points and dst_points must have shape (4,2).")


# -----------------  Filtering & Kernel Size Validation -----------------

def validate_kernel_size(kernel_size: int | list[int, ...]) -> None:
    """Validate kernel size for filtering."""
    # Handle single integer case
    if isinstance(kernel_size, int):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd integer.")
    # Handle tuple case
    elif isinstance(kernel_size, list):
        if not all(isinstance(x, int) and x > 0 and x % 2 == 1 for x in
                   kernel_size):
            raise ValueError(
                "Kernel size must be a tuple of positive odd integers.")
    else:
        raise TypeError(
            "Kernel size must be an integer or a tuple of integers.")


def validate_bilateral_filter_params(diameter: int, sigma_color: float,
                                     sigma_space: float) -> None:
    """Validate the parameters for the bilateral filter."""
    if not isinstance(diameter, int) or diameter <= 0 or diameter % 2 == 0:
        raise ValueError("Diameter must be an odd positive integer.")

    if not isinstance(sigma_color, (int, float)) or sigma_color <= 0:
        raise ValueError("Sigma color must be a positive number.")

    if not isinstance(sigma_space, (int, float)) or sigma_space <= 0:
        raise ValueError("Sigma space must be a positive number.")


def validate_2d_filter_kernel(kernel: np.ndarray) -> None:
    """Validate the kernel for a 2D convolution filter."""
    if not isinstance(kernel, np.ndarray):
        raise TypeError("Kernel must be a NumPy array.")

    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D matrix.")


# -----------------  Thresholding & Intensity Validation -----------------

def validate_threshold(threshold_lower: int, threshold_higher: int) -> None:
    """Validate that threshold values are positive integers and lower < higher."""
    if not (isinstance(threshold_lower, int | None)
            and isinstance(threshold_higher, int | None)):
        raise ValueError("Thresholds must be integers, or not define.")

    if threshold_lower is not None and threshold_higher is not None:
        if (threshold_lower < 0 or threshold_higher < 0
                or threshold_lower >= threshold_higher):
            raise ValueError(
                "Threshold values must be positive, and lower < higher.")


def validate_threshold_values(thresh: int, maxval: int) -> None:
    """Validate threshold and max value for thresholding."""
    if not isinstance(thresh, int):
        raise TypeError(f"Threshold must be an integer, got {type(thresh)}.")

    if not (0 <= thresh <= 255):
        raise ValueError(f"Threshold must be between 0 and 255, got {thresh}.")

    if not isinstance(maxval, int):
        raise TypeError(f"Max value must be an integer, got {type(maxval)}.")

    if not (0 <= maxval <= 255):
        raise ValueError(f"Max value must be between 0 and 255, got {maxval}.")


def validate_adaptive_threshold_params(max_value: int, block_size: int,
                                       c: int) -> None:
    """Validate parameters for adaptive thresholding."""
    if not isinstance(max_value, int) or not (0 <= max_value <= 255):
        raise ValueError("max_value must be an integer between 0 and 255.")

    if not isinstance(block_size,
                      int) or block_size % 2 == 0 or block_size <= 1:
        raise ValueError(
            "block_size must be an odd positive integer greater than 1.")

    if not isinstance(c, int):
        raise ValueError("C must be an integer.")


# -----------------  Resizing & Scaling Validation -----------------

def validate_resize_factors(factor_x: float, factor_y: float = None) -> None:
    """Validate the scaling factors for resizing."""
    if not isinstance(factor_x, (int, float)) or factor_x <= 0:
        raise ValueError("Scaling factor_x must be a positive number.")

    if factor_y is not None and (
            not isinstance(factor_y, (int, float)) or factor_y <= 0):
        raise ValueError("Scaling factor_y must be a positive number.")


# -----------------  Transformation & Rotation Validation -----------------

def validate_flip_type(flip_type: FlipType) -> None:
    """Validate the flip type for flipping an image."""
    if not isinstance(flip_type, FlipType):
        raise ValueError("flip_type must be an instance of FlipType Enum.")


def validate_rotation_type(rotate_type: RotateType) -> None:
    """Validate the rotation type."""
    if not isinstance(rotate_type, RotateType):
        raise ValueError("rotate_type must be an instance of RotateType Enum.")


def validate_clockwise_rotation(clockwise: bool) -> None:
    """Validate the rotation direction."""
    if not isinstance(clockwise, bool):
        raise ValueError("Rotation direction must be a boolean value.")


def validate_scale(scale: float) -> None:
    """Validate the scaling factor."""
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("Scale must be a positive number.")


# -----------------  Channel & Color Validation -----------------

def validate_channel_type(channel_type: ChannelType) -> None:
    """Validate if the provided channel is an instance of ChannelType Enum."""
    if not isinstance(channel_type, ChannelType):
        raise ValueError("Channel must be an instance of ChannelType Enum.")


def validate_bgr_channels(image: ImageType) -> None:
    """Validate if the image is a valid 3-channel (BGR) image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel (BGR) image.")

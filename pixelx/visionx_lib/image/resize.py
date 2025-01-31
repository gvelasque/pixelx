# pixelx/visionx_lib/image/resize.py

from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import cv2, ImageType


def resize_by_aspect_ratio(image: ImageType, new_width: int,
                           new_height: int | None = None,
                           dest: ImageType | None = None,
                           interpolation: int = cv2.INTER_LINEAR) -> ImageType:
    """Resize the image while maintaining its aspect ratio."""
    if new_width is not None:
        validations.validate_dimensions(new_width, 1)  # Width only

    if new_height is not None:
        validations.validate_dimensions(1, new_height)  # Height only

    # Calculate the aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]

    if new_width is not None:
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = int(new_height * aspect_ratio)

    return cv2.resize(image, dsize=(new_width, new_height), dst=dest,
                      interpolation=interpolation)


# TODO: CODE NOT USED
def resize_by_factor(image: ImageType, factor_x: float,
                     factor_y: float = None,
                     dest: ImageType | None = None,
                     interpolation: int = cv2.INTER_LINEAR) -> ImageType:
    """Resize the image by a scaling factor."""

    validations.validate_resize_factors(factor_x, factor_y)

    factor_y = factor_x if factor_y is None else factor_y

    return cv2.resize(image, dsize=None, dst=dest, fx=factor_x, fy=factor_y,
                      interpolation=interpolation)


def resize_by_width_height(image: ImageType, new_width: int,
                           new_height: int,
                           dest: ImageType | None = None,
                           interpolation: int = cv2.INTER_LINEAR) -> ImageType:
    """Resize the image to the specified width and height."""
    validations.validate_dimensions(new_width, new_height)

    return cv2.resize(image, dsize=(new_width, new_height), dst=dest,
                      interpolation=interpolation)

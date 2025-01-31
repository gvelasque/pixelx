# pixelx/visionx_lib/image/crop.py

from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import ImageType


def crop_image(image: ImageType, y_start: int, y_end: int,
               x_start: int, x_end: int) -> ImageType:
    """Crop the image using the specified coordinates."""
    validations.validate_coordinates(image, y_start, y_end, x_start, x_end)
    return image[y_start:y_end, x_start:x_end]

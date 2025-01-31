# pixelx/visionx_lib/image/roi_masks.py


from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import cv2, np, ImageType
from pixelx.visionx_lib.core.utils import get_image_dimensions
from pixelx.visionx_lib.core.enums import MaskType


def apply_triangular_mask(image: ImageType,
                          color: tuple[int, int, int]) -> ImageType:
    """Apply a triangular ROI mask to the image."""
    height, width = get_image_dimensions(image)
    validations.validate_dimensions(height, width)

    # Define the triangular region vertices
    vertices = np.array([[
        (int(width * 0.05), height),  # Bottom-left corner
        (int(width * 0.95), height),  # Bottom-right corner
        (int(width * 0.5), int(height * 0.55))  # Top-center
    ]], dtype=np.int32)

    # Ensure vertices array has the correct shape
    if vertices.shape != (1, 3, 2):
        raise ValueError("Vertices array must have shape (1, 3, 2).")

    # Create a blank single-channel mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill the ROI on the mask
    cv2.fillPoly(mask, vertices, color)

    return mask


def apply_rectangular_mask(image: ImageType,
                           start_point: tuple[int, int],
                           end_point: tuple[int, int],
                           color: tuple[int, int, int],
                           thickness: int) -> ImageType:
    """Apply a rectangular ROI mask to the image."""
    height, width = get_image_dimensions(image)
    validations.validate_dimensions(width, height)
    validations.validate_roi_inputs(start_point, end_point)

    # Create a blank single-channel mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw a rectangle
    cv2.rectangle(mask, start_point, end_point, color, thickness)

    return mask


def apply_circular_mask(image: ImageType, center: tuple[int, int],
                        radius: int, color: tuple[int, int, int],
                        thickness: int) -> ImageType:
    """Apply a circular ROI mask to the image."""
    height, width = get_image_dimensions(image)
    validations.validate_dimensions(width, height)
    validations.validate_roi_inputs(center=center, radius=radius)

    # Create a blank single-channel mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw a circle
    cv2.circle(mask, center, radius, color, thickness)

    return mask


def apply_roi_mask(image: ImageType,
                   mask_type: MaskType,
                   color: tuple[int, int, int],
                   thickness: int = 2,
                   center: tuple[int, int] | None = None,
                   radius: int | None = None,
                   start_point: tuple[int, int] | None = None,
                   end_point: tuple[int, int] | None = None) -> ImageType:
    """Applies a specified region of interest (ROI) mask to the image."""

    if mask_type == MaskType.TRIANGLE.value:
        return apply_triangular_mask(image, color)

    elif mask_type == MaskType.RECTANGULAR.value:
        validations.validate_roi_inputs(
            start_point=start_point, end_point=end_point)
        return apply_rectangular_mask(
            image, start_point, end_point, color, thickness)

    elif mask_type == MaskType.CIRCLE.value:
        validations.validate_roi_inputs(center=center, radius=radius)
        return apply_circular_mask(image, center, radius, color, thickness)

    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")

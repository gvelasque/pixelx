# pixelx/visionx_lib/image/transform.py

from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import cv2, np, ImageType
from pixelx.visionx_lib.core.enums import FlipType, RotateType


def flip_image(img: ImageType, flip_type: FlipType) -> ImageType:
    """Flip the image horizontally, vertically, or both."""
    validations.validate_flip_type(flip_type)
    return cv2.flip(img, flip_type.value)


def compute_new_bounding_box(rotation_matrix: ImageType, width: int,
                             height: int) -> tuple:
    """Compute the new bounding box dimensions after rotation."""
    validations.validate_transformation_matrix(rotation_matrix, (2, 3))
    validations.validate_dimensions(width, height)

    # Extract rotation components (cosine and sine of the rotation angle
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])

    # Compute new bounding box dimensions
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))

    return new_width, new_height


def rotate_center(img: ImageType, rotate_type: RotateType,
                  clockwise: bool = True, scale: float = 1.0) -> ImageType:
    """Rotate the image around its center by a specified angle."""
    validations.validate_rotation_type(rotate_type)
    validations.validate_clockwise_rotation(clockwise)
    validations.validate_scale(scale)

    # Get image dimensions
    (height, width) = img.shape[:2]
    center = (width // 2, height // 2)

    # Rotate the image
    angle = -rotate_type.value if clockwise else rotate_type.value

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Compute new bounding box
    new_width, new_height = compute_new_bounding_box(
        rotation_matrix, width, height)

    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Rotate image
    return cv2.warpAffine(img, rotation_matrix, (new_width, new_height))


def warp_perspective(img: ImageType, src_points: np.ndarray,
                     dst_points: np.ndarray) -> ImageType:
    """Apply perspective transformation to the image."""
    validations.validate_perspective_points(src_points, dst_points)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))


def apply_affine_transform(img: ImageType, matrix: np.ndarray,
                           output_size: tuple[int, int]) -> ImageType:
    """Apply an affine transformation to the image."""
    validations.validate_transformation_matrix(matrix, (2, 3))

    if not isinstance(output_size, tuple) or len(output_size) != 2:
        raise ValueError("output_size must be a tuple of (width, height).")

    if not all(isinstance(x, int) and x > 0 for x in output_size):
        raise ValueError("output_size values must be positive integers.")

    return cv2.warpAffine(img, matrix, output_size)

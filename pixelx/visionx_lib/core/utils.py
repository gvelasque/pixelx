# pixelx/visionx_lib/core/utils.py

from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import np


def get_image_dimensions(image: np.ndarray) -> tuple[int, int]:
    """Return height and width of an image."""
    return image.shape[:2]


def find_center_image(height: int | float,
                      width: int | float) -> tuple[int, int]:
    """Find the center of an image using its shape."""
    validations.validate_dimensions(int(width), int(height))
    return height // 2, width // 2


def split_line_coordinates(points: np.ndarray) -> tuple[list[int], list[int]]:
    """Split an array of point coordinates into x and y coordinate lists."""
    x_start, y_start, x_end, y_end = points
    return [x_start, x_end], [y_start, y_end]


def calculate_slope(line: np.ndarray) -> float:
    """Calculate the slope of a detected line segment."""
    if line.shape != (1, 4):
        raise ValueError("Line must be a NumPy array with shape (1, 4).")

    # Unpack the line coordinates and cast
    x1, y1, x2, y2 = line.flatten()  # Flattens the array safely
    h_dist = x2 - x1
    v_dist = y2 - y1

    # Return slope, handling vertical lines
    return np.divide(v_dist, h_dist, where=h_dist != 0,
                     out=np.full_like(h_dist, np.inf, dtype=float))

# pixelx/visionx_lib/lane/process_line.py

from pixelx.visionx_lib.core import utils
from pixelx.visionx_lib.core.base import np, ImageType


def separate_lines(
        image: ImageType, lines: np.ndarray,
        slope_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Separates detected lines into left and right lane lines based on slope."""

    # Return empty lists if no lines are detected
    if lines is None or len(lines) == 0:
        return np.empty((0, 4), dtype=int), np.empty((0, 4), dtype=int)

    left_lines, right_lines = [], []
    _, width = utils.get_image_dimensions(image)
    center_x = width // 2  # Center of the image

    for line in lines:
        # Unpack the line coordinates
        x1, y1, x2, y2 = line.flatten()
        slope = utils.calculate_slope(line)

        # Ignore near-horizontal lines
        if abs(slope) < slope_threshold:
            continue

        # Separate left and right lines
        if slope < 0 and x1 < center_x and x2 < center_x:
            left_lines.append([x1, y1, x2, y2])
        elif slope > 0 and x1 > center_x and x2 > center_x:
            right_lines.append([x1, y1, x2, y2])

    return np.array(left_lines), np.array(right_lines)


def extract_coordinates_from_lines(
        lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extracts all x and y coordinates from multiple detected lines."""
    all_x_coords, all_y_coords = [], []

    for line in lines:
        x_coords, y_coords = utils.split_line_coordinates(line)
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)

    return np.array(all_x_coords), np.array(all_y_coords)


def approximate_line(image: ImageType,
                     detected_lines: np.ndarray,
                     hidden_frac: float) -> tuple[int, ...] | None:
    """Fits a single line using linear regression."""
    if detected_lines is None or len(detected_lines) == 0:
        return None  # No sufficient lines detected

    height, width = utils.get_image_dimensions(image)

    # Extract x and y coordinates
    x_coords, y_coords = extract_coordinates_from_lines(detected_lines)

    if x_coords.size < 2 or y_coords.size < 2:
        raise None

    # Fit a line to the points (y = mx + b)
    # Reverse xy (Vertical lines)
    # TODO: update np.polyfit to np.polynomial.Polynomial.fit
    slope, intercept = np.polyfit(y_coords, x_coords, 1)

    # Define start and end points of the fitted line
    # Hidden fraction start from the top and extend to % of the image height
    y_start, y_end = height, int(height * hidden_frac)
    x_start = int(slope * y_start + intercept)
    x_end = int(slope * y_end + intercept)

    return x_start, y_start, x_end, y_end


def fit_lines(image: ImageType,
              left_lines: np.ndarray,
              right_lines: np.ndarray,
              hidden_frac: float = 0.6
              ) -> tuple[tuple[int, ...] | None, tuple[int, ...]] | None:
    """
    Approximates the left and right lane lines based on detected line segments.
    """
    left_line = approximate_line(image, left_lines, hidden_frac)
    right_line = approximate_line(image, right_lines, hidden_frac)
    return left_line, right_line

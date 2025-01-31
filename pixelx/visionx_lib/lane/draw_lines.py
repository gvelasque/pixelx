# pixelx/visionx_lib/lane/draw_line.py

from pixelx.visionx_lib.core.base import cv2, np, ImageType
from pixelx.visionx_lib.core.enums import to_bgr, MASK_COLORS


def draw_lines(
        image: ImageType, lines: list[np.ndarray],
        color: tuple[int, int, int] = MASK_COLORS["RED"],
        thickness: int = 2, in_place: bool = False) -> ImageType:
    """
    Draw lines on a blank image of the same shape as the input image or
    directly on the input image.
    """

    # If in_place is True, draw in place; otherwise, create a new blank image
    output_image = image if in_place else np.zeros_like(image, dtype=np.uint8)

    for line in lines:
        # Reshape the lines array to the format (N, 1, 2)
        # where N is the number of points in each line
        line_points = line.reshape(-1, 1, 2)

        # Draw the polyline on the output image
        cv2.polylines(img=output_image, pts=[line_points], isClosed=False,
                      color=to_bgr(color), thickness=thickness)
    return output_image


def draw_lane(image: ImageType, left_line: tuple[int, ...] | None,
              right_line: tuple[int, ...] | None, color: tuple[int, int, int],
              thickness: int, in_place: bool = False) -> ImageType:
    """Draws left and right lane lines on the image."""

    lines = []

    # Draw the left lane line
    if left_line is not None:
        left_start_x, left_start_y, left_end_x, left_end_y = left_line
        lines.append(np.array([[left_start_x, left_start_y],
                               [left_end_x, left_end_y]]))  # Left line

    if right_line is not None:
        right_start_x, right_start_y, right_end_x, right_end_y = right_line
        lines.append(np.array([[right_start_x, right_start_y],
                               [right_end_x, right_end_y]]))  # Right line

    return draw_lines(image, lines, color=color, thickness=thickness,
                      in_place=in_place)


def overlay_images(image: ImageType, overlay_image: ImageType,
                   alpha: float, beta: float, gamma: float) -> ImageType:
    return cv2.addWeighted(image, alpha, overlay_image, beta, gamma)

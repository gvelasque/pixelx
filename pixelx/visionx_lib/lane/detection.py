# pixelx/visionx_lib/lane/detection.py

from pixelx.visionx_lib.core.base import cv2, ImageType
from pixelx.visionx_lib.core.enums import MaskType
from pixelx.visionx_lib.image import apply_gaussian_blur, \
    convert_to_rgb2grayscale, apply_canny_edge_detection, load_image, \
    apply_roi_mask, resize_by_aspect_ratio, apply_detect_hough_lines, \
    display_image_plt
from pixelx.visionx_lib.lane.draw_lines import draw_lane
from pixelx.visionx_lib.lane.process_lines import separate_lines, fit_lines


def preprocess_image(image: str | ImageType, width: int,
                     height: int) -> tuple[ImageType, ImageType, ImageType]:
    """Load an image or a frame, and return the original, grayscale and resize."""
    # Load image
    original_image: ImageType = load_image(
        image) if isinstance(image, str) else image

    # Resize the image
    resize_image = resize_by_aspect_ratio(original_image, width, height)

    # Convert the image to grayscale
    grayscale_image = convert_to_rgb2grayscale(resize_image)

    return original_image, grayscale_image, resize_image


def detect_edges(image: ImageType, kernel_size: list[int, int],
                 deviation: float, threshold_lower: int, threshold_higher: int,
                 sigma: float, display: bool = False) -> ImageType:
    """Apply Gaussian blur followed by Canny edge detection."""
    # Apply Gaussian blur
    blurred = apply_gaussian_blur(image, kernel_size, deviation)

    # Apply Canny edge detection
    canny_edge = apply_canny_edge_detection(
        blurred, threshold_lower, threshold_higher, sigma)

    if display:
        display_image_plt(blurred, "Blurred", "gray")
        display_image_plt(canny_edge, "Canny edge", "gray")
    return canny_edge


def roi_mask(image: ImageType, mask_type: MaskType, color: tuple[int, int, int],
             thickness: int, center: tuple[int, int], radius: int,
             start_point: tuple[int, int], end_point: tuple[int, int],
             display: bool = False) -> ImageType:
    """Apply a region of interest (ROI) mask to focus on specific areas."""
    # Get the mask for the region of interest
    mask = apply_roi_mask(image, mask_type, color, thickness, center, radius,
                          start_point, end_point)

    # Apply the mask to the edges image
    roi = cv2.bitwise_and(image, mask)

    if display:
        display_image_plt(mask, "Mask", "gray")
        display_image_plt(roi, "Roi", "gray")
    return roi


def detect_lane(
        image: ImageType, edge_img: ImageType, rho: int,
        theta_degrees: float, threshold: int, min_line_length: int,
        max_line_gap: int, slope_threshold: float, hidden_frac: float
) -> tuple | tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """ Detects lane lines on the given image. """

    lines = apply_detect_hough_lines(
        edge_img, rho, theta_degrees, threshold, min_line_length, max_line_gap)

    if lines is None:
        return ()  # Return original if no lines detected

    # Separate left and right lines
    left_lines, right_lines = separate_lines(image, lines, slope_threshold)

    # Fit a single line for each side
    return fit_lines(image, left_lines, right_lines, hidden_frac)


def draw_detect_lane(original_image: ImageType, resize_image: ImageType,
                     detected_lines, color: tuple[int, int, int],
                     thickness: int, alpha: float, beta: float,
                     gamma: float) -> tuple[ImageType, ImageType]:
    """ Draw the detected lane on the original image."""
    # Extract the detected lane lines
    left_line, right_line = detected_lines

    # Create the lane overlay (mask)
    lane_overlay = draw_lane(
        resize_image, left_line, right_line, color, thickness)

    # Resize the lane overlay to match the original image dimensions
    resized_lane_overlay = resize_by_aspect_ratio(
        lane_overlay, original_image.shape[1])

    # Blend the resized lane overlay with the original image
    final_image = cv2.addWeighted(
        original_image, alpha, resized_lane_overlay, beta, gamma)

    return final_image, lane_overlay


def process_image(image: str | ImageType,
                  config_params: tuple[dict, ...],
                  display: bool = False) -> tuple[ImageType, ImageType]:
    """ Process an image to detect and visualize lanes. """

    (preprocess_config, edge_config, mask_config, detect_config,
     draw_config) = config_params

    # Load, resize and convert image to grayscale
    original_image, grayscale_image, resize_image = preprocess_image(
        image, **preprocess_config)

    # Apply edges detection
    edges = detect_edges(grayscale_image, **edge_config, display=display)

    # Get and apply roi mask
    masked_edges = roi_mask(edges, **mask_config, display=display)

    # Detect lane lines
    detected_lane = detect_lane(resize_image, masked_edges, **detect_config)

    # Draw lane lines and return overlay and mask image
    return draw_detect_lane(
        original_image, resize_image, detected_lane, **draw_config)

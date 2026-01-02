# oop_version/simple_lane_detection/lane_drawer.py

from pixelx.visionx_lib.core.base import cv2, ImageType
from pixelx.visionx_lib.image import resize_by_aspect_ratio
from pixelx.visionx_lib.lane.draw_lines import draw_lane


class LaneDrawer:
    """
    Handles all lane visualization operations including drawing lanes
    and overlaying them on the original image.
    """

    def __init__(
        self,
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0
    ):
        """
        Initialize the LaneDrawer with drawing parameters.
        
        Args:
            color: RGB color for lane lines
            thickness: Line thickness in pixels
            alpha: Weight for the original image in blending
            beta: Weight for the lane overlay in blending
            gamma: Scalar added to blended result
        """
        self.color = color
        self.thickness = thickness
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lane_overlay = None
        self.final_image = None

    def draw_lanes(
        self,
        original_image: ImageType,
        resize_image: ImageType,
        left_line: tuple[int, ...] | None,
        right_line: tuple[int, ...] | None
    ) -> tuple[ImageType, ImageType]:
        """
        Draw the detected lane lines on the image.
        
        Args:
            original_image: Original full-size image
            resize_image: Resized image used for detection
            left_line: Left lane line coordinates (x1, y1, x2, y2)
            right_line: Right lane line coordinates (x1, y1, x2, y2)
            
        Returns:
            Tuple of (final_image, lane_overlay)
        """
        # Create the lane overlay (mask)
        self.lane_overlay = draw_lane(
            resize_image, left_line, right_line, self.color, self.thickness
        )

        # Resize the lane overlay to match the original image dimensions
        resized_lane_overlay = resize_by_aspect_ratio(
            self.lane_overlay, original_image.shape[1]
        )

        # Blend the resized lane overlay with the original image
        self.final_image = cv2.addWeighted(
            original_image, self.alpha, resized_lane_overlay, self.beta, self.gamma
        )

        return self.final_image, self.lane_overlay

    def get_result(self) -> ImageType:
        """
        Get the final blended image with lane overlay.
        
        Returns:
            Final image with detected lanes drawn
        """
        return self.final_image

    def get_overlay(self) -> ImageType:
        """
        Get the lane overlay without the original image.
        
        Returns:
            Lane overlay image
        """
        return self.lane_overlay

# oop_version/simple_lane_detection/line_detector.py

from pixelx.visionx_lib.core.base import ImageType
from pixelx.visionx_lib.image import apply_detect_hough_lines
from pixelx.visionx_lib.lane.process_lines import separate_lines, fit_lines


class LineDetector:
    """
    Encapsulates all line detection logic including Hough line detection,
    line separation (left/right), and line fitting.
    """

    def __init__(
        self,
        rho: int = 1,
        theta_degrees: float = 180,
        threshold: int = 30,
        min_line_length: int = 30,
        max_line_gap: int = 20,
        slope_threshold: float = 0.5,
        hidden_frac: float = 0.65
    ):
        """
        Initialize the LineDetector with detection parameters.
        
        Args:
            rho: Distance resolution in pixels for Hough transform
            theta_degrees: Angle resolution in degrees for Hough transform
            threshold: Minimum number of votes for a line
            min_line_length: Minimum line length
            max_line_gap: Maximum gap between line segments
            slope_threshold: Minimum slope threshold for line filtering
            hidden_frac: Fraction of image height to hide from top
        """
        self.rho = rho
        self.theta_degrees = theta_degrees
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.slope_threshold = slope_threshold
        self.hidden_frac = hidden_frac
        self.detected_lines = None
        self.left_line = None
        self.right_line = None

    def detect_lines(
        self, image: ImageType, edge_img: ImageType
    ) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
        """
        Detect lane lines on the given image.
        
        Args:
            image: Original resized image
            edge_img: Edge-detected and masked image
            
        Returns:
            Tuple of (left_line, right_line) coordinates
        """
        # Apply Hough line detection
        self.detected_lines = apply_detect_hough_lines(
            edge_img,
            self.rho,
            self.theta_degrees,
            self.threshold,
            self.min_line_length,
            self.max_line_gap
        )

        if self.detected_lines is None:
            return None, None

        # Separate left and right lines
        left_lines, right_lines = separate_lines(
            image, self.detected_lines, self.slope_threshold
        )

        # Fit a single line for each side
        self.left_line, self.right_line = fit_lines(
            image, left_lines, right_lines, self.hidden_frac
        )

        return self.left_line, self.right_line

    def get_detected_lines(self) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
        """
        Get the most recently detected left and right lines.
        
        Returns:
            Tuple of (left_line, right_line) coordinates
        """
        return self.left_line, self.right_line

    def has_lines(self) -> bool:
        """
        Check if any lines were detected.
        
        Returns:
            True if at least one line was detected, False otherwise
        """
        return self.left_line is not None or self.right_line is not None

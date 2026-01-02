# oop_version/simple_lane_detection/image_processor.py

from pixelx.visionx_lib.core.base import cv2, ImageType
from pixelx.visionx_lib.image import (
    apply_gaussian_blur,
    convert_to_rgb2grayscale,
    apply_canny_edge_detection,
    load_image,
    apply_roi_mask,
    resize_by_aspect_ratio,
    display_image_plt
)


class ImageProcessor:
    """
    Handles all image preprocessing operations including loading, resizing,
    grayscale conversion, edge detection, and ROI masking.
    """

    def __init__(self, width: int = 640, height: int = None):
        """
        Initialize the ImageProcessor with target dimensions.
        
        Args:
            width: Target width for image resizing
            height: Target height for image resizing (optional)
        """
        self.width = width
        self.height = height
        self.original_image = None
        self.grayscale_image = None
        self.resize_image = None
        self.edges = None
        self.masked_edges = None

    def load_and_preprocess(self, image: str | ImageType) -> tuple[ImageType, ImageType, ImageType]:
        """
        Load an image or frame and return original, grayscale, and resized versions.
        
        Args:
            image: Path to image file or ImageType array
            
        Returns:
            Tuple of (original_image, grayscale_image, resize_image)
        """
        # Load image
        self.original_image = load_image(image) if isinstance(image, str) else image

        # Resize the image
        self.resize_image = resize_by_aspect_ratio(
            self.original_image, self.width, self.height
        )

        # Convert the image to grayscale
        self.grayscale_image = convert_to_rgb2grayscale(self.resize_image)

        return self.original_image, self.grayscale_image, self.resize_image

    def detect_edges(
        self,
        image: ImageType,
        kernel_size: list[int, int] = [5, 5],
        deviation: float = 0.0,
        threshold_lower: int = None,
        threshold_higher: int = None,
        sigma: float = 0.3,
        display: bool = False
    ) -> ImageType:
        """
        Apply Gaussian blur followed by Canny edge detection.
        
        Args:
            image: Input image (grayscale)
            kernel_size: Gaussian blur kernel size
            deviation: Gaussian blur standard deviation
            threshold_lower: Lower threshold for Canny edge detection
            threshold_higher: Upper threshold for Canny edge detection
            sigma: Sigma parameter for automatic threshold calculation
            display: Whether to display intermediate results
            
        Returns:
            Edge-detected image
        """
        # Apply Gaussian blur
        blurred = apply_gaussian_blur(image, kernel_size, deviation)

        # Apply Canny edge detection
        self.edges = apply_canny_edge_detection(
            blurred, threshold_lower, threshold_higher, sigma
        )

        if display:
            display_image_plt(blurred, "Blurred", "gray")
            display_image_plt(self.edges, "Canny edge", "gray")

        return self.edges

    def apply_roi_mask(
        self,
        image: ImageType,
        mask_type: str,  # String value like 'triangle', 'rectangular', 'circle'
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        center: tuple[int, int] = None,
        radius: int = None,
        start_point: tuple[int, int] = None,
        end_point: tuple[int, int] = None,
        display: bool = False
    ) -> ImageType:
        """
        Apply a region of interest (ROI) mask to focus on specific areas.
        
        Args:
            image: Input edge image
            mask_type: Type of mask to apply
            color: Mask color
            thickness: Mask line thickness
            center: Center point for circular masks
            radius: Radius for circular masks
            start_point: Start point for rectangle/triangle masks
            end_point: End point for rectangle/triangle masks
            display: Whether to display intermediate results
            
        Returns:
            Masked image with ROI applied
        """
        # Get the mask for the region of interest
        mask = apply_roi_mask(
            image, mask_type, color, thickness, center, radius,
            start_point, end_point
        )

        # Apply the mask to the edges image
        self.masked_edges = cv2.bitwise_and(image, mask)

        if display:
            display_image_plt(mask, "Mask", "gray")
            display_image_plt(self.masked_edges, "ROI", "gray")

        return self.masked_edges

    def get_processed_images(self) -> dict:
        """
        Get all processed image versions.
        
        Returns:
            Dictionary containing all processed image versions
        """
        return {
            'original': self.original_image,
            'grayscale': self.grayscale_image,
            'resized': self.resize_image,
            'edges': self.edges,
            'masked_edges': self.masked_edges
        }

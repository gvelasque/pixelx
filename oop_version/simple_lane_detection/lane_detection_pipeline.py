# oop_version/simple_lane_detection/lane_detection_pipeline.py

from pixelx.visionx_lib.core.base import ImageType
from image_processor import ImageProcessor
from line_detector import LineDetector
from lane_drawer import LaneDrawer


class LaneDetectionPipeline:
    """
    Main pipeline class that orchestrates the entire lane detection process.
    Integrates ImageProcessor, LineDetector, and LaneDrawer to provide a
    complete object-oriented solution for lane detection.
    """

    def __init__(self, config_params: tuple[dict, ...] = None):
        """
        Initialize the lane detection pipeline with configuration parameters.
        
        Args:
            config_params: Tuple of configuration dictionaries for each stage:
                          (preprocess_config, edge_config, mask_config, 
                           detect_config, draw_config)
        """
        if config_params:
            (preprocess_config, edge_config, mask_config, 
             detect_config, draw_config) = config_params
        else:
            # Default configurations
            preprocess_config = {'width': 640, 'height': None}
            edge_config = {
                'kernel_size': [5, 5], 'deviation': 0.0,
                'threshold_lower': None, 'threshold_higher': None, 'sigma': 0.3
            }
            mask_config = {
                'mask_type': 'triangle', 'color': (255, 255, 255), 'thickness': 2,
                'center': None, 'radius': None, 'start_point': None, 'end_point': None
            }
            detect_config = {
                'rho': 1, 'theta_degrees': 180, 'threshold': 30,
                'min_line_length': 30, 'max_line_gap': 20,
                'slope_threshold': 0.5, 'hidden_frac': 0.65
            }
            draw_config = {
                'color': (255, 0, 0), 'thickness': 5,
                'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0
            }

        # Initialize components
        self.image_processor = ImageProcessor(**preprocess_config)
        self.line_detector = LineDetector(**detect_config)
        self.lane_drawer = LaneDrawer(**draw_config)
        
        # Store configurations for edge detection and masking
        self.edge_config = edge_config
        self.mask_config = mask_config

    def process_image(
        self, image: str | ImageType, display: bool = False
    ) -> tuple[ImageType, ImageType]:
        """
        Process an image to detect and visualize lanes.
        
        Args:
            image: Path to image file or ImageType array
            display: Whether to display intermediate processing steps
            
        Returns:
            Tuple of (final_image, lane_overlay)
        """
        # Step 1: Load and preprocess the image
        original_image, grayscale_image, resize_image = \
            self.image_processor.load_and_preprocess(image)

        # Step 2: Detect edges
        edges = self.image_processor.detect_edges(
            grayscale_image, **self.edge_config, display=display
        )

        # Step 3: Apply ROI mask
        masked_edges = self.image_processor.apply_roi_mask(
            edges, **self.mask_config, display=display
        )

        # Step 4: Detect lane lines
        left_line, right_line = self.line_detector.detect_lines(
            resize_image, masked_edges
        )

        # Step 5: Draw lane lines and blend with original image
        final_image, lane_overlay = self.lane_drawer.draw_lanes(
            original_image, resize_image, left_line, right_line
        )

        return final_image, lane_overlay

    def get_components(self) -> dict:
        """
        Get access to individual pipeline components.
        
        Returns:
            Dictionary containing references to all pipeline components
        """
        return {
            'image_processor': self.image_processor,
            'line_detector': self.line_detector,
            'lane_drawer': self.lane_drawer
        }

    def reset(self):
        """
        Reset all component states for processing a new image.
        """
        self.image_processor = ImageProcessor(**{'width': self.image_processor.width, 
                                                   'height': self.image_processor.height})
        # Line detector and lane drawer maintain their configuration but will 
        # naturally reset when processing new images

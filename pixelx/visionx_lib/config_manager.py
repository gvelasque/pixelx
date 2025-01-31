# pixelx/visionx_lib/config_manager.py

from pixelx.visionx_lib.core.base import os, json

from pixelx.visionx_lib.core.enums import MASK_COLORS


class ConfigManager:
    """Handles loading and accessing configuration parameters."""

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self):
        """Load the configuration parameters from a JSON file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Configuration file {self.config_file} not found.")

        with open(self.config_file, "r") as file:
            return json.load(file)

    def get_params(self, section, key=None, default=None):
        """Retrieve a config value safely."""
        if key:
            return self.config.get(section, {}).get(key, default)
        return self.config.get(section, default)


def fetch_processing_params(config: ConfigManager):
    # Fetch preprocessing parameters
    preprocess_config: dict = config.get_params("image_processing",
                                                "preprocessing")

    # Fetch edge detection parameters
    edge_params = config.get_params("image_processing", "edge_detection")
    gaussian_blur_params = edge_params.get("gaussian_blur", {})
    canny_params = edge_params.get("canny", {})
    edge_config = {**gaussian_blur_params, **canny_params}

    # Fetch ROI mask parameters
    mask_config: dict = config.get_params("image_processing", "roi_mask")
    mask_config["color"] = MASK_COLORS.get(mask_config["color"].upper())

    # Fetch detect edges parameters
    detect_params = config.get_params("image_processing", "detect_lane")
    hough_lines_params = detect_params.get("hough_lines", {})
    separate_lines_params = detect_params.get("separate_lines", {})
    fit_lines_params = detect_params.get("fit_lines", {})
    detect_config: dict = {
        **hough_lines_params, **separate_lines_params, **fit_lines_params
    }

    # Fetch draw and detect lane parameters
    draw_params = config.get_params("image_processing", "draw_detect_lane")
    draw_lane_params = draw_params.get("draw_lane", {})
    draw_lane_params["color"] = MASK_COLORS.get(
        draw_lane_params["color"].upper())
    overlay_params = draw_params.get("overlay", {})
    draw_config: dict = {**draw_lane_params, **overlay_params}

    return (preprocess_config, edge_config, mask_config,
            detect_config, draw_config)

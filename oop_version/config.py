# pixelx/simple_lane_detection/config.py
import os
import logging

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_io_directories(base_path: str, io_types: list[str],
                         data_types: list[str]) -> dict:
    """
    Set up input and output directories for specified data types
    and I/O types, creating them if they don't exist.
    """
    base_dir = os.path.dirname(os.path.abspath(base_path))

    directories = {}

    for io_type in io_types:
        for data_type in data_types:
            dir_path = os.path.join(base_dir, "data", io_type, data_type)
            os.makedirs(dir_path, exist_ok=True)
            directories[f"{io_type}_{data_type}_dir"] = dir_path

    return directories


def setup_logging():
    """Configure the logging system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# pixelx/dir.py

from pixelx.visionx_lib.core.base import ImageType, logging, os
from pixelx.visionx_lib.image import save_image
from pixelx.visionx_lib.video.io import save_video_file


def get_files_in_directory(directory: str) -> dict[str, str]:
    """
    Returns a dictionary of file names and their full paths from the
    given directory.
    """
    logging.info(f"Get all files to directory: {directory}")
    try:
        return {file_name: os.path.join(directory, file_name) for file_name in
                os.listdir(directory)}
    except OSError as e:
        logging.error(f"Error accessing directory {directory}: {e}")
        return {}


def save_files_in_directory(
        directory: str, files: dict[str, ImageType | dict]) -> None:
    """Saves the provided files to the specified directory."""
    logging.info(f"Save processed files in directory: {directory}")
    try:
        for name, file in files.items():
            if isinstance(file, ImageType):
                image_path = os.path.join(directory, f"Processed_{name}")
                save_image(file, image_path)
                logging.info(f"Saving Processed_{name}")
            elif isinstance(file, dict) and "frames" in file:
                video_path = os.path.join(directory, f"Processed_{name}")
                save_video_file(
                    file["frames"], output_path=video_path,
                    fps=file["fps"], frame_size=file["frame_size"],
                    codec=file["codec"]
                )
                logging.info(f"Saving Processed_{name}")
    except OSError as e:
        logging.error(f"Error saving files in directory {directory}: {e}")


def process_directory(directory: str, process_function,
                      output_directory: str = None) -> None:
    """
    General function to process files in a directory using a provided
    function.
    """
    logging.info(f"Processing files in directory: {directory}")
    files = get_files_in_directory(directory)
    if not files:
        logging.info("No files to process.")
        return

    processed_files = {
        name: process_function(path) for name, path in files.items()}

    if output_directory is not None:
        save_files_in_directory(output_directory, files=processed_files)

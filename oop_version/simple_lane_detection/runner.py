# oop_version/simple_lane_detection/runner.py

from config import setup_logging, setup_io_directories
from pixelx.dir import process_directory
from pixelx.visionx_lib.config_manager import ConfigManager, fetch_processing_params
from pixelx.visionx_lib.core.base import ImageType, cv2, logging
from pixelx.visionx_lib.core.enums import VideoCodec
from pixelx.visionx_lib.image import display_image_cv2
from pixelx.visionx_lib.image.display_control import close_all_windows, handle_image_exit
from pixelx.visionx_lib.video.io import open_video_capture, get_frame_dimensions, get_frame_rate
from pixelx.visionx_lib.video.processing import process_video

from lane_detection_pipeline import LaneDetectionPipeline


def setup_logging_config():
    """Set up the application configuration."""
    setup_logging()


def configure_directories():
    """
    Set up input and output directories for images and videos,
    creating them if they don't exist.
    """
    return setup_io_directories(
        base_path=__file__,
        io_types=["input", "output"],
        data_types=["images", "videos"]
    )


def pipeline_image(image_path: str, display: bool = False) -> ImageType:
    """
    Process an image for lane detection using the OOP pipeline.
    
    Args:
        image_path: Path to the input image
        display: Whether to display the result
        
    Returns:
        Processed image with detected lanes
    """
    logging.info("Starting image processing with OOP pipeline...")
    try:
        # Initialize config
        config = ConfigManager("../simple_lane_detection/config.json")
        config_params = fetch_processing_params(config)

        # Create the OOP pipeline
        pipeline = LaneDetectionPipeline(config_params)

        # Process the image
        processed_image, _ = pipeline.process_image(image_path, display=False)

        if not display:
            return processed_image

        display_image_cv2(processed_image)
        handle_image_exit()
        close_all_windows()
        return processed_image
        
    except FileNotFoundError:
        logging.error("Image file not found. Please check the path.")
    except Exception as e:
        logging.exception(f"Unexpected error during image processing: {e}")
    finally:
        logging.info("Image processing complete.")


def process_frame_with_pipeline(image: ImageType, config_params: tuple) -> ImageType:
    """
    Wrapper function to process a single video frame.
    
    Args:
        image: Frame image to process
        config_params: Configuration parameters
        
    Returns:
        Processed frame with detected lanes
    """
    pipeline = LaneDetectionPipeline(config_params)
    processed_image, _ = pipeline.process_image(image, display=False)
    return processed_image


def pipeline_video(video_path: str, display: bool = False) -> dict:
    """
    Process a video for lane detection using the OOP pipeline.
    
    Args:
        video_path: Path to the input video
        display: Whether to display the result
        
    Returns:
        Dictionary containing processed frames and video metadata
    """
    logging.info("Starting video processing with OOP pipeline...")

    # Initialize config
    config = ConfigManager("../simple_lane_detection/config.json")
    config_params = fetch_processing_params(config)

    capture: cv2.VideoCapture | None = None
    try:
        capture = open_video_capture(video_path)
        if not capture:
            logging.error("Error: Could not open video file")
            return {}

        fps = get_frame_rate(capture)
        frame_size = get_frame_dimensions(capture)
        
        # Process video using the OOP approach
        processed_frames = process_video(
            capture,
            process_function=lambda img, cfg: LaneDetectionPipeline(cfg).process_image(img, display=False)[0],
            config_params=config_params,
            display=display
        )

        codec = VideoCodec.MP4V

        return {
            "frames": processed_frames,
            "fps": fps,
            "frame_size": frame_size,
            "codec": codec
        }

    except FileNotFoundError:
        logging.error("Video file not found. Please check the path.")
    except Exception as e:
        logging.exception(f"Unexpected error during video processing: {e}")
    finally:
        if capture:
            capture.release()
        logging.info("Video processing complete.")


if __name__ == "__main__":
    setup_logging_config()  # Set configuration of project

    directories = configure_directories()

    input_images_dir = directories["input_images_dir"]
    output_images_dir = directories["output_images_dir"]
    input_videos_dir = directories["input_videos_dir"]
    output_videos_dir = directories["output_videos_dir"]

    process_directory(
        input_images_dir,
        lambda path: pipeline_image(path, display=False),
        output_images_dir
    )

    process_directory(
        input_videos_dir,
        lambda path: pipeline_video(path, display=False),
        output_videos_dir
    )

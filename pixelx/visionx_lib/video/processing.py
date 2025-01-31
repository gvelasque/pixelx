# pixelx/visionx_lib/video/procession.py

from pixelx.visionx_lib.core.base import cv2, ImageType
from pixelx.visionx_lib.image.display_control import handle_image_exit
from pixelx.visionx_lib.video.io import get_frame_rate, read_frame, \
    display_frame, release_video_capture


def get_frame_skip_interval(fps: int, processing_rate: int) -> int:
    """Return the frame skip interval of the given video capture."""
    return max(1, fps // processing_rate)


def is_frame_processable(frame_count: int, skip_interval: int) -> bool:
    """Determine if this frame needs processing."""
    return (frame_count % skip_interval) == 0


def store_frame(frame, processed, processed_frames) -> None:
    """Stores a frame into the processed frames list."""
    processed_frames.append(frame if processed is None else processed)


def process_video(capture: cv2.VideoCapture, process_function: callable,
                  config_params: tuple[dict, ...], processing_rate: int = 10,
                  display: bool = False) -> list:
    """Process a video stream using a custom frame processing function."""

    try:
        fps = get_frame_rate(capture)
        skip_interval = get_frame_skip_interval(fps, processing_rate)
        frame_count: int = 0
        processed_frames: list[ImageType] = []

        while capture.isOpened():
            frame = read_frame(capture)

            if frame is None:
                print('Video capture failed or end of video reached')
                break

            # Determine if this frame needs processing
            processed_frame = None
            if is_frame_processable(frame_count, skip_interval):
                processed_frame, mask_lane = process_function(frame,
                                                              config_params)

            store_frame(frame, processed_frame, processed_frames)

            if display:
                display_frame(frame, processed_frame)

                # Handle keyboard interrupt (ESC key)
                if handle_image_exit(fps): break

            frame_count += 1
        return processed_frames
    except Exception as e:
        print(f"Error during video processing: {e}")
        return []

    finally:
        release_video_capture(capture, display)

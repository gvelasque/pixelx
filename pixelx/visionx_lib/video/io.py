# pixelx/visionx_lib/video/io.py

from pixelx.visionx_lib.core.base import cv2, ImageType
from pixelx.visionx_lib.core.enums import VideoCodec
from pixelx.visionx_lib.image import display_image_cv2
from pixelx.visionx_lib.image.display_control import close_all_windows


def open_video_capture(video_path: str) -> cv2.VideoCapture | None:
    """Opens a video file or camera stream for capture."""
    capture = cv2.VideoCapture(video_path)

    if capture.isOpened():
        return capture

    print('Error: Could not open video file or stream.')
    return None


def get_frame_dimensions(capture: cv2.VideoCapture) -> tuple[int, int]:
    """Return height and width of the given video capture object."""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height


def get_frame_rate(capture: cv2.VideoCapture) -> int:
    """Return the frame rate of the given video capture object."""
    return round(capture.get(cv2.CAP_PROP_FPS))


def get_fourcc(codec: VideoCodec) -> int:
    """Generate the FourCC code for the specified codec."""
    return cv2.VideoWriter.fourcc(*codec.value)


def read_frame(capture: cv2.VideoCapture) -> ImageType | None:
    """Reads a single frame from the video capture."""
    successfully_captured, frame = capture.read()
    return frame if successfully_captured else None


def display_frame(frame: ImageType, processed_frame: ImageType,
                  window_name: str = "Video") -> None:
    """Displays the current frame or processed frame."""
    to_display = frame if processed_frame is None else processed_frame
    display_image_cv2(to_display, window_name)


def create_video_writer(output_path: str, fps: int, frame_size: tuple[int, int],
                        codec: VideoCodec) -> cv2.VideoWriter:
    """Create VideoWriter object."""
    # Define the codec
    fourcc = get_fourcc(codec)

    # Create VideoWriter object
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def save_video_file(frames: list, output_path: str, fps: int,
                    frame_size: tuple[int, int], codec: VideoCodec) -> None:
    writer = create_video_writer(output_path, fps, frame_size, codec)

    for frame in frames:
        writer.write(frame)

    writer.release()
    print(f"Video saved to {output_path}")


# ----------
def release_video_capture(capture: cv2.VideoCapture, display: bool) -> None:
    """Releases video capture and closes any display windows."""
    capture.release()
    if display: close_all_windows()

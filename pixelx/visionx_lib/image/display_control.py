# pixelx/visionx_lib/image/display_control.py

from pixelx.visionx_lib.core.base import cv2


def wait_for_key_press(millisecond: int = 0) -> int:
    """Waits for a key press for a specified duration."""
    return cv2.waitKey(millisecond)


def handle_image_exit(millisecond: int = 0) -> bool:
    """Check if the ESC key (ASCII 27) was pressed to exit."""
    return wait_for_key_press(millisecond) & 0xFF == 27


def close_display_windows(window_name: str = "Image") -> None:
    """Destroy the specified window or all windows if no name is provided."""
    cv2.destroyWindow(window_name)


def close_all_windows() -> None:
    """Destroy all windows."""
    cv2.destroyAllWindows()

# pixelx/visionx_lib/core/enums.py

from enum import Enum


class FlipType(Enum):
    HORIZONTAL_FLIP = 1
    VERTICAL_FLIP = 0
    BOTH_FLIP = -1


class RotateType(Enum):
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270


class MaskType(Enum):
    TRIANGLE = "triangle"
    RECTANGULAR = "rectangular"
    CIRCLE = "circle"


MASK_COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 0, 255)
}


def to_bgr(rgb_color: tuple[int, int, int]) -> tuple:
    """Convert an RGB color tuple to a BGR color tuple."""
    return rgb_color[::-1]


class ChannelType(Enum):
    BLUE = 0
    GREEN = 1
    RED = 2


class VideoCodec(Enum):
    MP4V = 'mp4v'  # MPEG-4
    XVID = 'XVID'  # XviD
    MJPG = 'MJPG'  # Motion JPEG
    AVC1 = 'avc1'  # H.264/AVC
    H264 = 'H264'  # H.264
    DIVX = 'DIVX'  # DivX
    WMV1 = 'WMV1'  # Windows Media Video
    WMV2 = 'WMV2'  # Windows Media Video
    VP80 = 'VP80'  # VP8
    HEVC = 'HEVC'  # H.265/HEVC

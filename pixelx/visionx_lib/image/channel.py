# pixelx/visionx_lib/image/channel.py

from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import ImageType
from pixelx.visionx_lib.core.enums import ChannelType


def extract_channel(image: ImageType, channel_type: ChannelType) -> ImageType:
    """Extract a specific channel from a BGR image."""
    validations.validate_bgr_channels(image)
    validations.validate_channel_type(channel_type)
    return image[:, :, channel_type.value]

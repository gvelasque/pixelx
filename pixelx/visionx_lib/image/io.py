# pixelx/visionx_lib/image/io.py

from pixelx.visionx_lib.image import convert_bgr2rgb
from pixelx.visionx_lib.core import validations
from pixelx.visionx_lib.core.base import cv2, Path, ImageType, plt


def load_image(image_path: str) -> ImageType:
    """Load an image from a file path."""
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("Input image path must be a non-empty string.")

    # Load the image
    image = cv2.imread(image_path)

    # Validate image
    validations.validate_image(image, image_path)

    return image


def save_image(image: ImageType, output_path: str) -> bool:
    """Save an image to a file path."""
    validations.validate_image(image, output_path)

    # Ensure the output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(output_path, image)


def display_image_cv2(image: ImageType, window_name: str = "Image") -> None:
    """Display the image using cv2."""
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, image.shape[1], image.shape[0])
    cv2.imshow(window_name, image)


def display_image_plt(image: ImageType, window_name: str = "Image",
                      cmap: str | None = None) -> None:
    """Displays an image using Matplotlib."""
    # image = convert_bgr2rgb(image)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(window_name)
    plt.show()


def display_images_plt(images: list, cmap=None) -> None:
    """Displays images using Matplotlib."""
    plt.figure(figsize=(20, 20))
    for i, image in enumerate(images):
        image = convert_bgr2rgb(image)
        plt.subplot(3, 2, i + 1)
        plt.imshow(image, cmap)
        plt.autoscale(tight=True)
        # plt.axis('off')
        plt.title(f"Image {i + 1}")
    plt.show()

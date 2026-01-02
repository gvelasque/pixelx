# Procedural Version - Simple Lane Detection

## Overview
This directory contains the **original procedural implementation** of the simple lane detection system. It follows a functional programming approach where lane detection is accomplished through a series of function calls that process the image step-by-step.

## Implementation Approach

### Procedural Programming Style
The procedural version organizes the code as a sequence of functions, each performing a specific task:

1. **`preprocess_image()`** - Loads and prepares the image
2. **`detect_edges()`** - Applies Gaussian blur and Canny edge detection
3. **`roi_mask()`** - Applies region of interest masking
4. **`detect_lane()`** - Detects lane lines using Hough transform
5. **`draw_detect_lane()`** - Draws detected lanes on the original image

### Key Characteristics
- ✅ **Sequential Execution**: Functions are called in a specific order
- ✅ **Stateless Functions**: Each function operates independently
- ✅ **Simple and Direct**: Easy to understand for simple use cases
- ✅ **Minimal Abstraction**: Direct manipulation of data

### File Structure
```
procedural_version/
└── simple_lane_detection/
    ├── runner.py          # Main entry point
    ├── config.json        # Configuration parameters
    └── data/              # Input/output directories
        ├── input/
        │   ├── images/
        │   └── videos/
        └── output/
            ├── images/
            └── videos/
```

## How It Works

The main processing pipeline in `runner.py` calls the `process_image()` function from `pixelx.visionx_lib.lane.detection`, which:

1. Loads and preprocesses the image
2. Converts to grayscale and resizes
3. Applies edge detection
4. Masks the region of interest
5. Detects lines using Hough transform
6. Fits and draws the detected lanes

## Usage

### Running the Procedural Version
```bash
cd procedural_version/simple_lane_detection
python runner.py
```

### Processing a Single Image
```python
from pixelx.visionx_lib.lane.detection import process_image
from pixelx.visionx_lib.config_manager import ConfigManager, fetch_processing_params

config = ConfigManager("config.json")
config_params = fetch_processing_params(config)
processed_image, overlay = process_image("path/to/image.jpg", config_params)
```

## Advantages of Procedural Approach
- **Simplicity**: Easy to follow the control flow
- **Quick Prototyping**: Fast to write and test
- **Minimal Overhead**: No class instantiation or object management
- **Suitable for Scripts**: Perfect for one-off processing tasks

## Limitations
- **Scalability**: Difficult to extend with new features
- **Code Reusability**: Hard to reuse individual components
- **State Management**: No encapsulation of intermediate results
- **Testing**: Harder to unit test individual components
- **Maintainability**: Changes can affect multiple parts of the code

## Configuration

All processing parameters are defined in `config.json`:
- Image preprocessing settings (width, height)
- Edge detection parameters (Gaussian blur, Canny thresholds)
- ROI mask configuration
- Hough transform parameters
- Drawing and overlay settings

## Dependencies
- OpenCV (cv2)
- NumPy
- Matplotlib
- VisualX library (pixelx.visionx_lib)

---

For a more modular and maintainable approach, see the **OOP version** in the `oop_version/` directory.

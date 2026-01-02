# Object-Oriented Version - Simple Lane Detection

## Overview
This directory contains the **object-oriented (OOP) implementation** of the simple lane detection system. It demonstrates modern software engineering principles by organizing code into well-defined classes with clear responsibilities.

## Implementation Approach

### Object-Oriented Programming Style
The OOP version structures the code using classes that encapsulate data and behavior:

1. **`ImageProcessor`** - Handles all image preprocessing operations
2. **`LineDetector`** - Encapsulates line detection logic
3. **`LaneDrawer`** - Manages lane visualization
4. **`LaneDetectionPipeline`** - Orchestrates the entire detection process

### Key Characteristics
- ✅ **Encapsulation**: Related data and methods are grouped together
- ✅ **Modularity**: Each class has a single, well-defined responsibility
- ✅ **Reusability**: Components can be used independently
- ✅ **Maintainability**: Changes are isolated to specific classes
- ✅ **Testability**: Easy to unit test individual components
- ✅ **Extensibility**: Simple to add new features or modify behavior

### File Structure
```
oop_version/
└── simple_lane_detection/
    ├── runner.py                    # Main entry point
    ├── lane_detection_pipeline.py   # Main pipeline orchestrator
    ├── image_processor.py           # Image preprocessing class
    ├── line_detector.py             # Line detection class
    ├── lane_drawer.py               # Lane drawing class
    ├── config.json                  # Configuration parameters
    └── data/                        # Input/output directories
        ├── input/
        │   ├── images/
        │   └── videos/
        └── output/
            ├── images/
            └── videos/
```

## Class Architecture

### 1. ImageProcessor Class
**Purpose**: Handles all image preprocessing tasks

**Key Methods**:
- `load_and_preprocess(image)` - Loads and prepares images
- `detect_edges(image, ...)` - Applies edge detection
- `apply_roi_mask(image, ...)` - Applies region of interest masking
- `get_processed_images()` - Returns all intermediate results

**Benefits**:
- Maintains state of processed images
- Provides easy access to intermediate results
- Encapsulates preprocessing logic

### 2. LineDetector Class
**Purpose**: Encapsulates line detection algorithms

**Key Methods**:
- `detect_lines(image, edge_img)` - Main detection method
- `get_detected_lines()` - Returns detected line coordinates
- `has_lines()` - Checks if lines were detected

**Benefits**:
- Configurable detection parameters
- Maintains detection state
- Easy to swap detection algorithms

### 3. LaneDrawer Class
**Purpose**: Manages lane visualization

**Key Methods**:
- `draw_lanes(original, resized, left, right)` - Draws lanes
- `get_result()` - Returns final blended image
- `get_overlay()` - Returns lane overlay only

**Benefits**:
- Separates drawing logic from detection
- Configurable appearance
- Maintains drawing state

### 4. LaneDetectionPipeline Class
**Purpose**: Orchestrates the complete detection workflow

**Key Methods**:
- `process_image(image, display)` - Main processing pipeline
- `get_components()` - Access to individual components
- `reset()` - Resets pipeline state

**Benefits**:
- Single entry point for lane detection
- Coordinates all components
- Provides high-level interface

## How It Works

The OOP version processes images through a pipeline of objects:

```python
# Initialize the pipeline
pipeline = LaneDetectionPipeline(config_params)

# Process an image
final_image, overlay = pipeline.process_image("path/to/image.jpg")
```

Internally:
1. **ImageProcessor** loads and preprocesses the image
2. **ImageProcessor** detects edges and applies masking
3. **LineDetector** identifies lane lines
4. **LaneDrawer** visualizes the results

## Usage

### Running the OOP Version
```bash
cd oop_version/simple_lane_detection
python runner.py
```

### Using the Pipeline Programmatically
```python
from lane_detection_pipeline import LaneDetectionPipeline
from pixelx.visionx_lib.config_manager import ConfigManager, fetch_processing_params

# Load configuration
config = ConfigManager("config.json")
config_params = fetch_processing_params(config)

# Create pipeline
pipeline = LaneDetectionPipeline(config_params)

# Process image
final_image, overlay = pipeline.process_image("path/to/image.jpg")
```

### Using Individual Components
```python
from image_processor import ImageProcessor
from line_detector import LineDetector
from lane_drawer import LaneDrawer

# Create components with custom settings
processor = ImageProcessor(width=800, height=600)
detector = LineDetector(threshold=50, min_line_length=40)
drawer = LaneDrawer(color=(0, 255, 0), thickness=8)

# Use them independently
original, grayscale, resized = processor.load_and_preprocess("image.jpg")
edges = processor.detect_edges(grayscale, ...)
masked = processor.apply_roi_mask(edges, ...)
left, right = detector.detect_lines(resized, masked)
final, overlay = drawer.draw_lanes(original, resized, left, right)
```

## Advantages of OOP Approach
- **Modularity**: Each class handles a specific responsibility
- **Reusability**: Components can be used in different contexts
- **Maintainability**: Changes are localized to specific classes
- **Testability**: Easy to unit test each component independently
- **Extensibility**: Simple to add new features (e.g., curved lane detection)
- **State Management**: Objects maintain their own state
- **Code Organization**: Clear structure and relationships

## Comparison with Procedural Version

| Aspect | Procedural | OOP |
|--------|-----------|-----|
| **Code Organization** | Sequential functions | Classes with methods |
| **State Management** | No state | Encapsulated state |
| **Reusability** | Limited | High |
| **Testability** | Harder | Easier |
| **Extensibility** | Difficult | Simple |
| **Learning Curve** | Low | Moderate |
| **Best For** | Simple scripts | Large projects |

## Future Extensions

The OOP architecture makes it easy to add:
- **Curved lane detection**: Extend `LineDetector` with polynomial fitting
- **Multiple lane detection**: Modify `LineDetector` to find more than 2 lanes
- **Custom visualizations**: Create new drawer classes
- **Alternative algorithms**: Swap components without affecting others
- **Performance monitoring**: Add logging to each component
- **Caching**: Store intermediate results for efficiency

## Configuration

All processing parameters are defined in `config.json`, identical to the procedural version for fair comparison.

## Dependencies
- OpenCV (cv2)
- NumPy
- Matplotlib
- VisualX library (pixelx.visionx_lib)

---

For the original procedural implementation, see the `procedural_version/` directory.

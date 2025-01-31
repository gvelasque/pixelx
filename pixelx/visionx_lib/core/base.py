# pixelx/visionx_lib/core/base.py

import cv2
import numpy as np
import json
import logging
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import TypeAlias

# Define a common ImageType alias for NumPy images
ImageType: TypeAlias = np.ndarray

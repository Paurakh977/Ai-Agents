"""
Utility functions for image handling
"""

import os
from .constants import IMAGE_DIR


def ensure_image_directory_exists():
    """
    Ensure that the images directory exists, creating it if necessary.

    Returns:
        str: Path to the images directory
    """
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Created images directory at {IMAGE_DIR}")
    
    return IMAGE_DIR 
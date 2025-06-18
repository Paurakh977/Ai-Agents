"""
Utility functions for image handling
"""

import os
from .constants import IMAGE_DIR, GENERATED_IMAGES_DIR


def ensure_image_directory_exists():
    """
    Ensure that the images directory exists, creating it if necessary.

    Returns:
        str: Path to the images directory
    """
    # Create main images directory if it doesn't exist
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Created images directory at {IMAGE_DIR}")
    
    # Also ensure the generated images directory exists
    if not os.path.exists(GENERATED_IMAGES_DIR):
        os.makedirs(GENERATED_IMAGES_DIR)
        print(f"Created generated images directory at {GENERATED_IMAGES_DIR}")
    
    return IMAGE_DIR 
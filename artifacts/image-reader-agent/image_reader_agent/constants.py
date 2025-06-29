"""
Constants for the Image Reader Agent.
"""

import os

# Models to use for the agent
GEMINI_MODEL = "gemini-2.0-flash-exp"
GEMINI_IMAGE_GENERATION_MODEL = "gemini-2.0-flash-preview-image-generation"

# Image storage paths
IMAGE_DIR = "images"
GENERATED_IMAGES_DIR = os.path.join(IMAGE_DIR, "generated") 
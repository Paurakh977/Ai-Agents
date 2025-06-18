"""
Tool for analyzing images using Google's multimodal model
"""

import os
from typing import Dict, Optional

import google.genai.types as types
from google.adk.tools.tool_context import ToolContext

from ..constants import IMAGE_DIR
from ..utils import ensure_image_directory_exists


def analyze_image(
    tool_context: ToolContext,
    image_id: Optional[str] = None,
) -> Dict:
    """
    Analyze an image using the multimodal capabilities of Gemini.
    If image_id is not provided, will use the most recently uploaded image.

    Args:
        tool_context: ADK tool context
        image_id: Optional identifier for a specific image

    Returns:
        Dictionary with analysis results
    """
    try:
        # Get the current image from state
        if image_id:
            image_to_analyze = image_id
        elif "current_image" in tool_context.state:
            image_to_analyze = tool_context.state["current_image"]
        else:
            return {
                "status": "error",
                "message": "No image found to analyze. Please upload an image first.",
            }

        # The image has already been loaded as an artifact in the before_model_callback
        # so we just need to load it for analysis
        try:
            # This will make the image available to the multimodal model
            tool_context.load_artifact_sync(filename=image_to_analyze)
            
            # Return success along with image metadata
            return {
                "status": "success",
                "message": f"Image {image_to_analyze} loaded for analysis.",
                "image": image_to_analyze,
                "notes": "The image is now available for the Gemini model to analyze directly.",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error loading image as artifact: {str(e)}",
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error analyzing image: {str(e)}",
        } 
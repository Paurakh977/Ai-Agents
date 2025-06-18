"""
Tool for analyzing images using Google's multimodal model
"""

import os
import asyncio
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

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

        print(f"[Image Analysis] Analyzing image: {image_to_analyze}")
        
        # Look for the image file directly
        possible_paths = [
            image_to_analyze,  # Direct path
            os.path.join(IMAGE_DIR, image_to_analyze),  # In images dir
            os.path.join(IMAGE_DIR, "generated", image_to_analyze),  # In generated dir
        ]
        
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            return {
                "status": "error",
                "message": f"Image file '{image_to_analyze}' not found",
            }
            
        # Read the image and create a new artifact
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Determine mime type from file extension
            extension = os.path.splitext(image_path)[1].lower()
            if extension == ".jpg" or extension == ".jpeg":
                mime_type = "image/jpeg"
            elif extension == ".png":
                mime_type = "image/png"
            elif extension == ".gif":
                mime_type = "image/gif"
            else:
                mime_type = f"image/{extension[1:]}"  # Remove the dot
                
            # Create the artifact
            image_artifact = types.Part(
                inline_data=types.Blob(data=image_data, mime_type=mime_type)
            )
            
            # Define an async function to handle all async operations
            async def save_load_attach_artifact():
                try:
                    # Save the artifact asynchronously
                    artifact_id = await tool_context.save_artifact(filename=image_to_analyze, artifact=image_artifact)
                    print(f"[Image Analysis] Saved image as artifact: {image_to_analyze} (id: {artifact_id})")
                    
                    # Load the artifact for the model
                    await tool_context.load_artifact(filename=image_to_analyze)
                    print(f"[Image Analysis] Loaded artifact for model: {image_to_analyze}")
                    
                    # Attach the artifact to next response
                    await tool_context.attach_artifact_to_next_response(image_to_analyze)
                    print(f"[Image Analysis] Attached artifact to next response: {image_to_analyze}")
                    
                    return artifact_id
                except Exception as e:
                    print(f"[Image Analysis] Error in async operation: {str(e)}")
                    return None
            
            # Create a new loop and run the async function in a separate thread
            loop = asyncio.new_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            
            # Run the async operations in a separate thread
            future = executor.submit(lambda: loop.run_until_complete(save_load_attach_artifact()))
            artifact_id = future.result()
            
            # Clean up
            executor.shutdown(wait=True)
            loop.close()
            
            # Check if the artifact was handled successfully
            if artifact_id:
                # Return success along with image metadata
                return {
                    "status": "success",
                    "message": f"Image {image_to_analyze} loaded for analysis.",
                    "image": image_to_analyze,
                    "image_displayed": True,
                    "notes": "The image is now available for the Gemini model to analyze directly.",
                }
            else:
                return {
                    "status": "partial_success",
                    "message": f"Image loaded but could not be displayed properly.",
                    "image": image_to_analyze,
                    "image_displayed": False,
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating artifact from image file: {str(e)}",
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error analyzing image: {str(e)}",
        } 
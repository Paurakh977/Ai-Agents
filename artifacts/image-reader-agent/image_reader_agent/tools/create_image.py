"""
Tool for creating and editing images using Google's Gemini image generation model.
"""

import os
import uuid
import asyncio
from typing import Dict, Optional

import google.genai as genai
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from ..constants import IMAGE_DIR, GEMINI_IMAGE_GENERATION_MODEL


def create_image(
    tool_context: ToolContext,
    prompt: str,
    edit_image_id: Optional[str] = None,
) -> Dict:
    """
    Create or edit an image using Gemini's image generation model.

    Args:
        tool_context: ADK tool context
        prompt: The prompt to generate an image from
        edit_image_id: ID of an existing image to edit (optional)

    Returns:
        Dictionary with results
    """
    try:
        print(f"[Image Creation] Starting image {'editing' if edit_image_id else 'generation'}")
        print(f"[Image Creation] Prompt: {prompt}")
        if edit_image_id:
            print(f"[Image Creation] Editing image: {edit_image_id}")

        # Initialize the GenAI client
        client = genai.Client()

        # Configure generation parameters
        generation_config = genai_types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        )

        if edit_image_id:
            # EDITING MODE: Find the image to edit
            # Try different possible locations
            image_path = None
            image_data = None

            # Look in various locations
            possible_paths = [
                edit_image_id,  # Direct path
                os.path.join(IMAGE_DIR, edit_image_id),  # In images dir
                os.path.join(IMAGE_DIR, "generated", edit_image_id),  # In generated subdir
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break

            # If image wasn't found by path, try loading from artifact
            if not image_path:
                try:
                    # Try to get image data from artifacts
                    artifact = tool_context.get_artifact(filename=edit_image_id)
                    if artifact and hasattr(artifact, "inline_data") and artifact.inline_data:
                        image_data = artifact.inline_data.data
                        mime_type = artifact.inline_data.mime_type
                        print(f"[Image Creation] Retrieved image data from artifact: {edit_image_id}")
                except Exception as e:
                    print(f"[Image Creation] Error retrieving artifact: {str(e)}")

            # If we still don't have the image, return error
            if not image_path and not image_data:
                return {
                    "status": "error",
                    "message": f"Image '{edit_image_id}' not found for editing",
                }

            # Prepare image for editing
            try:
                if image_path:
                    # Read image from file
                    with open(image_path, "rb") as f:
                        image_data = f.read()
                    print(f"[Image Creation] Loaded image from file: {image_path}")
                    # Get mime type from file extension
                    extension = os.path.splitext(image_path)[1].lower()
                    if extension == ".jpg" or extension == ".jpeg":
                        mime_type = "image/jpeg"
                    else:
                        mime_type = f"image/{extension[1:]}"  # Remove the dot

                # Create image part for Gemini
                image_part = genai_types.Part.from_bytes(
                    data=image_data, mime_type=mime_type
                )

                # Generate content with both text prompt and image
                response = client.models.generate_content(
                    model=GEMINI_IMAGE_GENERATION_MODEL,
                    contents=[prompt, image_part],
                    config=generation_config,
                )

            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error processing image for editing: {str(e)}",
                }
        else:
            # GENERATION MODE: Create image from text prompt only
            try:
                response = client.models.generate_content(
                    model=GEMINI_IMAGE_GENERATION_MODEL,
                    contents=prompt,
                    config=generation_config,
                )
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error generating image: {str(e)}",
                }

        # Extract generated image and any text response
        image_data = None
        response_text = ""

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # Extract text response
                if hasattr(part, "text") and part.text:
                    response_text += part.text
                # Extract image data
                elif hasattr(part, "inline_data") and part.inline_data:
                    if hasattr(part.inline_data, "data") and part.inline_data.data:
                        image_data = part.inline_data.data
                        mime_type = getattr(part.inline_data, "mime_type", "image/png")

        # Check if we got an image
        if not image_data:
            return {
                "status": "error",
                "message": f"No image was generated by the model. Response: {response_text or 'No text response.'}",
            }

        # Create a unique filename for the image
        operation_type = "edited" if edit_image_id else "generated"
        # Create a hash value to use as part of the filename for uniqueness
        hash_value = abs(hash(prompt + str(uuid.uuid4()))) % 10000
        filename = f"{operation_type}_image_{hash_value}.png"

        # Ensure "images/generated" directory exists
        from ..utils import ensure_image_directory_exists
        image_dir = ensure_image_directory_exists()
        generated_dir = os.path.join(image_dir, "generated")
        if not os.path.exists(generated_dir):
            os.makedirs(generated_dir)

        # Save the image to disk
        filepath = os.path.join(generated_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_data)
        print(f"[Image Creation] Saved image to: {filepath}")

        # Save as an artifact to make it available to the model
        try:
            # Create an artifact without display_name
            image_artifact = genai_types.Part(
                inline_data=genai_types.Blob(
                    data=image_data,
                    mime_type=mime_type
                )
            )
            
            # Save the artifact
            tool_context.save_artifact_sync(
                filename=filename, artifact=image_artifact
            )
            print(f"[Image Creation] Saved image as artifact: {filename}")
            
            # Return success along with all relevant information
            return {
                "status": "success",
                "message": f"Successfully {operation_type} image based on the prompt",
                "image_path": filepath,
                "image_filename": filename,
                "response_text": response_text,
            }
        except Exception as e:
            # If artifact saving fails, still return success with the image path
            return {
                "status": "partial_success",
                "message": f"Image {operation_type} but could not save as artifact: {str(e)}",
                "image_path": filepath,
                "image_filename": filename,
                "response_text": response_text,
            }

    except Exception as e:
        # Catch-all for any unexpected errors
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}",
        }
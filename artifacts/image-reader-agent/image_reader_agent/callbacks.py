"""
Callbacks for the Image Reader Agent.
"""

import os
import asyncio
from typing import Dict, Optional

import google.genai.types as types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

from .constants import IMAGE_DIR
from .utils import ensure_image_directory_exists


async def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Callback that executes before the model is called.
    Detects and saves inline images from user messages.

    Args:
        callback_context: The callback context
        llm_request: The LLM request

    Returns:
        Optional[LlmResponse]: None to allow normal processing
    """
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    print(f"[Image Callback] Processing for agent: {agent_name} (Inv: {invocation_id})")

    # IMPORTANT: Clean ALL images in the conversation history to remove display_name attributes
    # This prevents the "display_name parameter is not supported in Gemini API" error
    if llm_request.contents:
        for content in llm_request.contents:
            if hasattr(content, "parts"):
                for part in content.parts:
                    if (hasattr(part, "inline_data") and part.inline_data and
                        hasattr(part.inline_data, "mime_type") and 
                        part.inline_data.mime_type and
                        part.inline_data.mime_type.startswith("image/")):
                        
                        # Remove display_name if present as it's not supported by Gemini API
                        if hasattr(part.inline_data, "display_name"):
                            print(f"[Image Callback] Removing display_name from conversation history image")
                            delattr(part.inline_data, "display_name")

    # Ensure image directory exists
    image_dir = ensure_image_directory_exists()

    # Get the last user message parts
    last_user_message_parts = []
    if llm_request.contents and llm_request.contents[-1].role == "user":
        if llm_request.contents[-1].parts:
            last_user_message_parts = llm_request.contents[-1].parts

    print(f"[Image Callback] User message parts count: {len(last_user_message_parts)}")

    # Process any image parts we found
    image_count = 0
    latest_image_name = None

    for part in last_user_message_parts:
        # Debug info
        print(f"[Image Callback] Examining part type: {type(part)}")

        # Make sure it's an image with mime type and data
        if not hasattr(part, "inline_data") or not part.inline_data:
            continue

        mime_type = getattr(part.inline_data, "mime_type", None)
        if not mime_type or not mime_type.startswith("image/"):
            continue

        image_data = getattr(part.inline_data, "data", None)
        if not image_data:
            continue
        
        # IMPORTANT: Remove display_name if present - this is causing the Gemini API error
        if hasattr(part.inline_data, "display_name"):
            print(f"[Image Callback] Removing display_name attribute that's not supported by Gemini API")
            delattr(part.inline_data, "display_name")

        # We have an image to save
        image_count += 1
        print(f"[Image Callback] Found image #{image_count}")

        # Get the file extension from mime type
        extension = mime_type.split("/")[-1]
        if extension == "jpeg":
            extension = "jpg"

        # Generate simple sequential filename
        image_name = f"uploaded_image_{image_count}.{extension}"
        image_path = os.path.join(image_dir, image_name)
        latest_image_name = image_name  # Track the latest image

        # Save the image
        try:
            print(f"[Image Callback] Saving image to: {image_path}")
            with open(image_path, "wb") as f:
                f.write(image_data)
            print(f"[Image Callback] Saved image: {image_name}")
            
            # Create artifact for the model to access
            image_artifact = types.Part(
                inline_data=types.Blob(data=image_data, mime_type=mime_type)
            )
            
            # Save as an artifact and attach to next response - since we're in an async function, 
            # we can use await directly
            try:
                # Since we're already in an async function, we can use await directly
                artifact_id = await callback_context.save_artifact(
                    filename=image_name, artifact=image_artifact
                )
                print(f"[Image Callback] Saved image as artifact: {image_name} (id: {artifact_id})")
                
                # CRITICAL: Mark the artifact to be displayed in the model's response
                await callback_context.attach_artifact_to_next_response(image_name)
                print(f"[Image Callback] Attached artifact to response: {image_name}")
                
                # Store the current image in state for future reference
                callback_context.state["current_image"] = image_name
                callback_context.state["current_image_path"] = image_path
            except Exception as e:
                print(f"[Image Callback] Error saving image as artifact: {str(e)}")
                
        except Exception as e:
            print(f"[Image Callback] Error saving image: {str(e)}")

    # Log the total number of images processed
    if image_count > 0:
        print(f"[Image Callback] Saved {image_count} images to {image_dir}")

    # Continue with normal execution
    return None 
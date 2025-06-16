"""
Callbacks for the Image Reader Agent.
"""

import os
from typing import Dict, Optional

import google.genai.types as types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

from .constants import IMAGE_DIR
from .utils import ensure_image_directory_exists


def before_model_callback(
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

        # Save the image
        try:
            print(f"[Image Callback] Saving image to: {image_path}")
            with open(image_path, "wb") as f:
                f.write(image_data)
            print(f"[Image Callback] Saved image: {image_name}")
            
            # Save image as artifact for the model to access
            image_artifact = types.Part(
                inline_data=types.Blob(data=image_data, mime_type=mime_type)
            )
            
            # Save as an artifact
            try:
                artifact_version = callback_context.tool_context.save_artifact(
                    filename=image_name, artifact=image_artifact
                )
                print(f"[Image Callback] Saved image as artifact: {image_name} (version {artifact_version})")
                
                # Store the current image in state
                callback_context.tool_context.state["current_image"] = image_name
                callback_context.tool_context.state["current_image_version"] = artifact_version
            except Exception as e:
                print(f"[Image Callback] Error saving image as artifact: {str(e)}")
                
        except Exception as e:
            print(f"[Image Callback] Error saving image: {str(e)}")

    # Log the total number of images processed
    if image_count > 0:
        print(f"[Image Callback] Saved {image_count} images to {image_dir}")

    # Continue with normal execution
    return None 
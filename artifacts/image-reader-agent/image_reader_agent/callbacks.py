"""
Callbacks for the Image Reader Agent.
"""

import os
import asyncio
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

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
                            
    # Check and record available artifacts at session start
    try:
        # This will examine what artifacts are currently available from previous model runs
        print("[Image Callback] Checking for available artifacts")
        artifacts = []
        
        # Use ThreadPoolExecutor to run the async function
        def run_async(async_func):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_func())
            finally:
                loop.close()
                
        # Define artifact listing function
        async def list_artifacts_async():
            return await callback_context.list_artifacts()
        
        # Execute the async operation
        with ThreadPoolExecutor(max_workers=1) as executor:
            artifacts = executor.submit(run_async, list_artifacts_async).result() or []
        
        if artifacts:
            print(f"[Image Callback] Current artifacts: {artifacts}")
            # Store available artifacts in state for reference
            callback_context.state["available_artifacts"] = artifacts
            
            # For convenience, track most recent image artifact
            if artifacts:
                latest_artifact = artifacts[-1]
                callback_context.state["last_artifact_filename"] = latest_artifact
                print(f"[Image Callback] Latest artifact: {latest_artifact}")
    except Exception as e:
        print(f"[Image Callback] Error checking artifacts: {str(e)}")

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
            
            # Save and attach artifact using ThreadPoolExecutor
            try:
                # Define async functions for artifact operations
                async def save_artifact_async():
                    artifact_id = await callback_context.save_artifact(
                        filename=image_name, artifact=image_artifact
                    )
                    print(f"[Image Callback] Saved image as artifact: {image_name} (id: {artifact_id})")
                    return artifact_id
                
                async def attach_artifact_async(filename):
                    await callback_context.attach_artifact_to_next_response(filename)
                    print(f"[Image Callback] Attached artifact to response: {filename}")
                
                # Run the async operations in sequence
                with ThreadPoolExecutor(max_workers=1) as executor:
                    # Save artifact first
                    artifact_id = executor.submit(run_async, save_artifact_async).result()
                    
                    # Then attach it
                    if artifact_id is not None:
                        executor.submit(run_async, lambda: attach_artifact_async(image_name)).result()
                
                # Store key information in state
                callback_context.state["current_image"] = image_name
                callback_context.state["current_image_path"] = image_path
                callback_context.state["last_artifact_id"] = artifact_id
                callback_context.state["last_artifact_filename"] = image_name
                callback_context.state["last_uploaded_image"] = image_name
                
                # Also mark with a timestamp to track upload sequence
                import time
                callback_context.state[f"uploaded_time_{image_name}"] = time.time()
                
            except Exception as e:
                print(f"[Image Callback] Error handling artifact: {str(e)}")
                
        except Exception as e:
            print(f"[Image Callback] Error saving image: {str(e)}")

    # Log the total number of images processed
    if image_count > 0:
        print(f"[Image Callback] Saved {image_count} images to {image_dir}")

    # Continue with normal execution
    return None 
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
            image_path = None
            image_data = None
            
            # First check if we have it in the state
            state_image_path = tool_context.state.get("current_image_path")
            state_image_name = tool_context.state.get("current_image")
            
            # Look for the image file path
            if edit_image_id == state_image_name and state_image_path and os.path.exists(state_image_path):
                image_path = state_image_path
                print(f"[Image Creation] Found image in state: {image_path}")
            else:
                # Try loading from artifacts first
                try:
                    # Define an async function to load the artifact
                    async def load_artifact_async():
                        try:
                            artifact = await tool_context.load_artifact(filename=edit_image_id)
                            if artifact and artifact.inline_data and artifact.inline_data.data:
                                print(f"[Image Creation] Successfully loaded artifact: {edit_image_id}")
                                return artifact.inline_data.data, artifact.inline_data.mime_type
                            return None, None
                        except Exception as e:
                            print(f"[Image Creation] Error loading artifact: {str(e)}")
                            return None, None
                    
                    # Run the async function in a new event loop
                    loop = asyncio.new_event_loop()
                    artifact_data, mime_type = loop.run_until_complete(load_artifact_async())
                    loop.close()
                    
                    if artifact_data:
                        # We found the image in artifacts, use it directly
                        image_data = artifact_data
                        print(f"[Image Creation] Using image from artifact: {edit_image_id}")
                    else:
                        # Try different possible file locations
                        possible_paths = [
                            edit_image_id,  # Direct path
                            os.path.join(IMAGE_DIR, edit_image_id),  # In images dir
                            os.path.join(IMAGE_DIR, "generated", edit_image_id),  # In generated subdir
                        ]

                        for path in possible_paths:
                            if os.path.exists(path):
                                image_path = path
                                break

                except Exception as e:
                    print(f"[Image Creation] Error while trying to load artifact: {str(e)}")
                    # Continue to try file paths

            # If image wasn't found by artifact or path
            if not image_path and not image_data:
                # Try to list available artifacts to help diagnose the issue
                async def list_artifacts_async():
                    try:
                        artifacts = await tool_context.list_artifacts()
                        print(f"[Image Creation] Available artifacts: {artifacts}")
                    except Exception as e:
                        print(f"[Image Creation] Error listing artifacts: {str(e)}")
                
                loop = asyncio.new_event_loop()
                loop.run_until_complete(list_artifacts_async())
                loop.close()
                
                return {
                    "status": "error",
                    "message": f"Image '{edit_image_id}' not found for editing. Please provide the exact filename.",
                    "available_images": tool_context.state.get("current_image", "None")
                }

            # If we have a path but no data yet, load the image data
            if image_path and not image_data:
                try:
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
                    
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error reading image file: {str(e)}",
                    }
                
            # Make sure we have image data by now
            if not image_data:
                return {
                    "status": "error",
                    "message": "Could not load the image data for editing.",
                }

            # Prepare image for editing
            try:
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
        
        # Update state to track the image
        tool_context.state["current_image"] = filename
        tool_context.state["current_image_path"] = filepath
        tool_context.state["image_generated"] = True  # Track that we've generated an image

        # Create an artifact
        image_artifact = genai_types.Part(
            inline_data=genai_types.Blob(
                data=image_data,
                mime_type=mime_type
            )
        )
        
        # Define an async function to handle all artifact operations
        async def save_and_attach_artifact():
            try:
                # Save the artifact
                artifact_id = await tool_context.save_artifact(filename=filename, artifact=image_artifact)
                print(f"[Image Creation] Saved artifact: {filename} (id: {artifact_id})")
                
                # CRITICAL: Explicitly attach the artifact to be displayed in the model's response
                await tool_context.attach_artifact_to_next_response(filename)
                print(f"[Image Creation] Attached artifact to response: {filename}")
                
                # List all artifacts to confirm
                artifacts = await tool_context.list_artifacts()
                print(f"[Image Creation] All artifacts: {artifacts}")
                
                return artifact_id
            except Exception as e:
                print(f"[Image Creation] Error in artifact operations: {str(e)}")
                return None
        
        # Run the async operations
        try:
            # Use ThreadPoolExecutor to run the async function
            from concurrent.futures import ThreadPoolExecutor
            
            def run_async(async_func):
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(async_func())
                finally:
                    new_loop.close()
            
            # Execute the async operations
            with ThreadPoolExecutor(max_workers=1) as executor:
                artifact_id = executor.submit(run_async, save_and_attach_artifact).result()
            
            # Return success response with artifact info
            if artifact_id is not None:
                # Important: Update state with this artifact info for future edits
                tool_context.state["last_artifact_id"] = artifact_id
                tool_context.state["last_artifact_filename"] = filename
                
                return {
                    "status": "success",
                    "message": f"Successfully {operation_type} image based on the prompt",
                    "image_path": filepath,
                    "image_filename": filename,
                    "artifact_id": artifact_id,
                    "image_displayed": True,
                    "response_text": response_text,
                }
            else:
                # Still succeed even if artifact handling failed
                return {
                    "status": "partial_success",
                    "message": f"Image {operation_type} but could not save as artifact",
                    "image_path": filepath,
                    "image_filename": filename,
                    "image_displayed": False,
                    "response_text": response_text,
                }
                
        except Exception as e:
            # If artifact handling fails, still return success with the image path
            print(f"[Image Creation] Exception in executor: {str(e)}")
            return {
                "status": "partial_success",
                "message": f"Image {operation_type} but could not save as artifact: {str(e)}",
                "image_path": filepath,
                "image_filename": filename,
                "image_displayed": False,
                "response_text": response_text,
            }

    except Exception as e:
        # Catch-all for any unexpected errors
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}",
        }
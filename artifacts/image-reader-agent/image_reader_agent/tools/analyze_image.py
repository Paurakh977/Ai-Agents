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
        # Get the current image information from state
        current_image = tool_context.state.get("current_image")
        current_image_path = tool_context.state.get("current_image_path")
        last_uploaded_image = tool_context.state.get("last_uploaded_image")
        
        # Determine which image to analyze
        image_path = None
        image_data = None
        image_filename = None
        
        # If specific image_id is provided, use that
        if image_id:
            image_filename = image_id
            print(f"[Image Analysis] Analyzing specific image: {image_id}")
            
            # First try to find by id as path
            if os.path.exists(image_id):
                image_path = image_id
                print(f"[Image Analysis] Using direct path: {image_path}")
            
            # Check if it matches the current image
            elif image_id == current_image and current_image_path and os.path.exists(current_image_path):
                image_path = current_image_path
                print(f"[Image Analysis] Using current image path: {image_path}")
            
            # Check standard locations
            else:
                possible_paths = [
                    image_id,  # Direct path
                    os.path.join(IMAGE_DIR, image_id),  # In images dir
                    os.path.join(IMAGE_DIR, "generated", image_id),  # In generated subdir
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        image_path = path
                        print(f"[Image Analysis] Found image at: {image_path}")
                        break
                
                # If still not found, try loading from artifacts
                if not image_path:
                    print(f"[Image Analysis] Image not found by path, trying artifacts")
                    
                    # Define an async function to load the artifact
                    async def load_artifact_async():
                        try:
                            artifact = await tool_context.load_artifact(filename=image_id)
                            if artifact and artifact.inline_data and artifact.inline_data.data:
                                print(f"[Image Analysis] Successfully loaded artifact: {image_id}")
                                return artifact.inline_data.data, artifact.inline_data.mime_type
                            return None, None
                        except Exception as e:
                            print(f"[Image Analysis] Error loading artifact: {str(e)}")
                            return None, None
                    
                    # Run the async function using ThreadPoolExecutor
                    def run_async(async_func):
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(async_func())
                        finally:
                            new_loop.close()
                    
                    # Execute the async loading operation
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        artifact_data, mime_type = executor.submit(run_async, load_artifact_async).result()
                    
                    if artifact_data:
                        # We found the image in artifacts, use it directly
                        image_data = artifact_data
                        print(f"[Image Analysis] Using image from artifact: {image_id}")
                    else:
                        # List available artifacts for debugging
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            def list_artifacts_func():
                                async def list_artifacts_async():
                                    try:
                                        artifacts = await tool_context.list_artifacts()
                                        print(f"[Image Analysis] Available artifacts: {artifacts}")
                                        return artifacts
                                    except Exception as e:
                                        print(f"[Image Analysis] Error listing artifacts: {str(e)}")
                                        return []
                                return run_async(list_artifacts_async)
                            
                            available_artifacts = executor.submit(list_artifacts_func).result() or []
                            tool_context.state["available_artifacts"] = available_artifacts
        
        # Use most recent image if no specific image provided or specified one not found
        if not image_path and not image_data:
            # First: try most recently uploaded image if it exists
            if last_uploaded_image and not image_id:
                print(f"[Image Analysis] Using last uploaded image: {last_uploaded_image}")
                upload_path = os.path.join(IMAGE_DIR, last_uploaded_image)
                if os.path.exists(upload_path):
                    image_path = upload_path
                    image_filename = last_uploaded_image
            # Second: try current image in state
            elif not image_path and current_image and current_image_path and os.path.exists(current_image_path):
                image_path = current_image_path
                image_filename = current_image
                print(f"[Image Analysis] Using current image from state: {image_path}")
            else:
                if not image_id:
                    print("[Image Analysis] No image_id provided, searching for images")
                else:
                    print(f"[Image Analysis] Could not find image: {image_id}, searching for alternatives")
                
                # Look for any image in the images directory
                ensure_image_directory_exists()
                all_images = []
                
                # First prioritize uploaded images
                for file in os.listdir(IMAGE_DIR):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        if file.startswith("uploaded_"):
                            # Higher priority for uploaded images (newer timestamp)
                            all_images.append((os.path.join(IMAGE_DIR, file), 
                                              os.path.getmtime(os.path.join(IMAGE_DIR, file)) + 1000))
                        else:
                            all_images.append((os.path.join(IMAGE_DIR, file), 
                                              os.path.getmtime(os.path.join(IMAGE_DIR, file))))
                
                # Check in generated subdirectory if it exists
                generated_dir = os.path.join(IMAGE_DIR, "generated")
                if os.path.exists(generated_dir):
                    for file in os.listdir(generated_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                            all_images.append((os.path.join(generated_dir, file), 
                                             os.path.getmtime(os.path.join(generated_dir, file))))
                
                # Find most recent image
                if all_images:
                    # Sort by modification time (most recent first)
                    all_images.sort(key=lambda x: x[1], reverse=True)
                    image_path = all_images[0][0]
                    image_filename = os.path.basename(image_path)
                    print(f"[Image Analysis] Using most recent image: {image_path}")
                else:
                    return {
                        "status": "error",
                        "message": "No images found to analyze. Please upload or generate an image first.",
                    }
        
        # By now we should have image_path or image_data
        if not image_path and not image_data:
            return {
                "status": "error",
                "message": "Could not find any image to analyze.",
            }
            
        # If we only have the path, read the image data
        if image_path and not image_data:
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                print(f"[Image Analysis] Read image data from: {image_path}")
                
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
        
        # Update state with current image info
        tool_context.state["current_image"] = image_filename
        if image_path:
            tool_context.state["current_image_path"] = image_path
        
        # Create image Part for the analysis
        image_part = types.Part(
            inline_data=types.Blob(
                data=image_data,
                mime_type=mime_type
            )
        )
        
        # Also save and attach the artifact for display in the model response
        async def save_load_attach_artifact():
            try:
                # Save as artifact for future reference
                artifact_id = await tool_context.save_artifact(filename=image_filename, artifact=image_part)
                print(f"[Image Analysis] Saved artifact: {image_filename} (id: {artifact_id})")
                
                # Attach to response to ensure it shows in the UI
                await tool_context.attach_artifact_to_next_response(image_filename)
                print(f"[Image Analysis] Attached artifact to response: {image_filename}")
                
                # Store artifact info in state for future reference
                tool_context.state["last_artifact_id"] = artifact_id
                tool_context.state["last_artifact_filename"] = image_filename
                
                return artifact_id
            except Exception as e:
                print(f"[Image Analysis] Error in artifact operations: {str(e)}")
                return None
                
        # Run the async artifact operations
        try:
            # Use ThreadPoolExecutor to run the async function
            def run_async(async_func):
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(async_func())
                finally:
                    new_loop.close()
            
            # Execute the async operations
            with ThreadPoolExecutor(max_workers=1) as executor:
                artifact_id = executor.submit(run_async, save_load_attach_artifact).result()
                
            print(f"[Image Analysis] Artifact operations completed, ID: {artifact_id}")
        except Exception as e:
            print(f"[Image Analysis] Error in artifact threading: {str(e)}")
            # Continue even if artifact handling failed
        
        # Tell the user what image we're analyzing
        return {
            "status": "success",
            "message": f"Analyzing image: {image_filename}",
            "image_path": image_path,
            "image_filename": image_filename,
            "image_artifact_id": tool_context.state.get("last_artifact_id")
        }

    except Exception as e:
        # Catch all other errors
        return {"status": "error", "message": f"Error analyzing image: {str(e)}"} 
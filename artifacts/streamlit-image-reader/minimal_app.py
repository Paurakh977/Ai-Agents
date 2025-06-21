"""
Minimal Streamlit + ADK Image Reader App
"""

import os
import streamlit as st
from dotenv import load_dotenv
import asyncio
from typing import Dict, Optional, List, Union, Any, Literal, overload
import base64
import logging
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ADK components
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.artifacts import InMemoryArtifactService

# Load environment variables
load_dotenv()

# Configure API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("API_KEY not found in .env file")
    st.stop()

# Constants
APP_NAME = "image_reader"
USER_ID = "streamlit_user"
GEMINI_MODEL = "gemini-2.0-flash-exp"
IMAGE_DIR = "images"

# Create images directory if it doesn't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

# Define analyze_image tool
async def analyze_image(tool_context: ToolContext, *, image_index: Optional[int] = None, file_name: Optional[str] = None) -> Dict[str, Any]:
    """Tool to analyze an image with Gemini's multimodal capabilities.
    
    Args:
        tool_context: ToolContext object
        image_index: Integer, the 1-based index of the image in the uploaded images list
        file_name: String, the exact filename of the image to analyze
        
    Returns:
        dict: A dictionary containing the analysis results, mime type, and image metadata
    """
    try:
        logger.info(f"Analyze image tool called with image_index={image_index}, file_name={file_name}")
        
        # Verify that exactly one parameter is provided
        if (image_index is None and file_name is None) or (image_index is not None and file_name is not None):
            error_msg = "You must provide EITHER image_index OR file_name, but not both and not neither"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get list of available images from state
        available_images = tool_context.state.get("all_uploaded_images", [])
        image_count = len(available_images)
        logger.info(f"Available images in state: {image_count} images")
        logger.info(f"Image filenames: {available_images}")
        
        if not available_images:
            logger.warning("No images found in state")
            return {"error": "No images available to analyze"}
            
        # Determine which image to analyze based on parameters
        image_to_analyze = None
        
        if file_name is not None:
            # Check if the specified file name exists in available images
            if file_name in available_images:
                image_to_analyze = file_name
                logger.info(f"Analyzing image specified by file_name: {file_name}")
            else:
                logger.warning(f"Requested file '{file_name}' not found in available images")
                return {"error": f"Image '{file_name}' not found in uploaded images. Available images: {available_images}"}
        
        elif image_index is not None:
            # Convert to 0-based index for internal use
            idx = image_index - 1
            if 0 <= idx < len(available_images):
                image_to_analyze = available_images[idx]
                logger.info(f"Analyzing image at index {image_index} (file: {image_to_analyze})")
            else:
                logger.warning(f"Image index {image_index} out of range (1-{len(available_images)})")
                return {"error": f"Image index {image_index} out of range. Available images: 1-{len(available_images)}"}
        
        # Load the artifact for the model
        if not image_to_analyze:
            logger.error("Failed to determine which image to analyze")
            return {"error": "Failed to determine which image to analyze"}
            
        logger.info(f"Attempting to load artifact: {image_to_analyze}")
        
        try:
            
            # Load the artifact
            artifact = await tool_context.load_artifact(filename=image_to_analyze)
            logger.info(f"Successfully loaded artifact: {image_to_analyze} (mime_type: {artifact.inline_data.mime_type})")
            
            # Update current image in state
            tool_context.state["last_analyzed_image"] = image_to_analyze
            tool_context.state["last_analyzed_index"] = available_images.index(image_to_analyze) + 1
            
            # Return successful result
            return {
                "success": f"artifact {image_to_analyze} loaded for analysis",
                "mime_type": artifact.inline_data.mime_type,
                # "data": artifact.inline_data.data,
                "image_name": image_to_analyze,
                "image_index": available_images.index(image_to_analyze) + 1,
                "total_images": len(available_images),
                
            }
        except Exception as e:
            logger.error(f"Failed to load artifact {image_to_analyze}: {str(e)}")
            return {"error": f"Failed to load image '{image_to_analyze}': {str(e)}"}
        
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        return {"error": f"Error: {str(e)}"}

# Define callback to process uploaded images
async def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """Process images in user messages and maintain proper state for multiple images."""
    
    # Extract user message parts
    if not llm_request.contents or llm_request.contents[-1].role != "user":
        logger.info("No user message found in callback")
        return None
        
    user_parts = llm_request.contents[-1].parts if llm_request.contents[-1].parts else []
    logger.info(f"Found {len(user_parts)} parts in user message")
    
    # Initialize state variables if they don't exist - CRITICAL: Use get() with default to preserve existing state
    if "all_uploaded_images" not in callback_context.state:
        callback_context.state["all_uploaded_images"] = []
    if "current_images" not in callback_context.state:
        callback_context.state["current_images"] = []
    if "image_versions" not in callback_context.state:
        callback_context.state["image_versions"] = {}
    
    # IMPORTANT: Get existing images to preserve state across calls
    existing_images = callback_context.state.get("all_uploaded_images", [])
    logger.info(f"Existing images in state before processing: {len(existing_images)}")
    
    # Track images in this message
    images_in_message = []
    image_count = 0
    
    # Look for image parts
    for i, part in enumerate(user_parts):
        if not hasattr(part, "inline_data") or not part.inline_data:
            continue
            
        if not getattr(part.inline_data, "mime_type", "").startswith("image/"):
            continue
            
        image_data = getattr(part.inline_data, "data", None)
        if not image_data:
            continue
            
        # Found an image
        image_count += 1
        mime_type = part.inline_data.mime_type
        extension = mime_type.split("/")[-1]
        if extension == "jpeg":
            extension = "jpg"
            
        # Generate unique image name with timestamp and UUID
        timestamp = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())[:8]
        image_name = f"uploaded_image_{timestamp}_{unique_id}.{extension}"
        logger.info(f"Processing image {image_count}: {image_name} (mime_type: {mime_type})")
        
        # Save as artifact
        image_artifact = genai_types.Part(
            inline_data=genai_types.Blob(
                data=image_data,
                mime_type=mime_type
            )
        )
        
        # Save artifact
        try:    
            artifact_version = await callback_context.save_artifact(
                filename=image_name, 
                artifact=image_artifact
            )
            logger.info(f"Successfully saved artifact: {image_name} (version: {artifact_version})")
            
            # Track this image in message
            images_in_message.append(image_name)
            
            # CRITICAL: Add to all uploaded images if not already there (preserve existing)
            if image_name not in callback_context.state["all_uploaded_images"]:
                callback_context.state["all_uploaded_images"].append(image_name)
                
            # Track version information
            callback_context.state["image_versions"][image_name] = artifact_version
            
            logger.info(f"Added image to state - name: {image_name}")
            
        except Exception as e:
            logger.error(f"Failed to save artifact {image_name}: {str(e)}")
    
    # Update state with images from this message
    if images_in_message:
        callback_context.state["current_images"] = images_in_message
        total_images = len(callback_context.state["all_uploaded_images"])
        logger.info(f"Updated state with {len(images_in_message)} new images")
        logger.info(f"Total images in session: {total_images}")
            
        # Create helpful messages about available images for state
        image_list_str = ""
        for i, img in enumerate(callback_context.state["all_uploaded_images"]):
            image_list_str += f"{i+1}. {img}\n"
        callback_context.state["image_list"] = image_list_str.strip()
        callback_context.state["total_images"] = total_images
        
        # Update state with a descriptive summary for the agent
        image_summary = f"You have access to {total_images} images."
        callback_context.state["image_summary"] = image_summary
        logger.info(f"Image summary: {image_summary}")
    
    return None

# Create agent with dynamic instruction
def create_agent():
    """Create the image reader agent with dynamic state and artifact placeholders"""
    logger.info("Creating image reader agent")
    return Agent(
        name="image_reader_agent",
        description="Analyzes images uploaded by users",
        model=GEMINI_MODEL,
        before_model_callback=before_model_callback,
        tools=[analyze_image],
        instruction="""
        # Image Analysis Agent
        
        You analyze images that users upload and provide detailed descriptions.
        
        ## Available Images
        
        {image_summary?}
        
        Total images uploaded in this session: {total_images?}
        
        {image_list?}
        
        Last analyzed image: {last_analyzed_image?}
        
        ## Your Responsibilities
        
        When a user uploads an image:
        1. Describe what you see in detail
        2. If text is visible, transcribe it
        3. Answer any questions about the image content
        
        ## Using Multiple Images
        
        If multiple images have been uploaded:
        - The user can refer to specific images by number (e.g., "analyze the 3rd image")
        - You MUST use the analyze_image tool with EXACTLY ONE of these parameters:
          - image_index: The 1-based index of the image (e.g. 1, 2, 3) - PREFERRED METHOD
          - file_name: The exact filename of the image (use only if you know the filename)
        
        ## Tool Usage Examples
        
        ALWAYS use this format to analyze images:
        
        ```
        analyze_image(image_index=3)
        ```
        
        The image_index parameter counts from 1, not 0. So the first image is 1, second is 2, etc.
        
        If user asks about a specific image by number (e.g., "tell me about the 2nd image"), 
        ALWAYS call analyze_image with the appropriate image_index.
        
        NEVER call analyze_image() without parameters - you must always specify which image to analyze!
        
        Always be helpful and accurate in your analysis.
        """
    )

# Initialize app
st.title("📷 Simple Image Analyzer")
st.write("Upload an image or ask questions about the last uploaded image")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = f"{os.urandom(4).hex()}"
    
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Create shared session service, artifact service and agent
@st.cache_resource
def get_services_and_agent():
    """Create services and agent only once"""
    agent = create_agent()
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    return session_service, artifact_service, agent

# Initialize session management
async def ensure_session_exists(session_service, session_id):
    """Create a session if it doesn't exist"""
    try:
        session = await session_service.get_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            session_id=session_id
        )
        if not session:
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
            return True  # New session created
        return False  # Existing session found
    except Exception as e:
        logger.error(f"Session error: {str(e)}")
        st.error(f"Session error: {str(e)}")
        return False

# Process agent response
async def process_agent_response(runner, content, session_id):
    """Run the agent and process the response"""
    try:
        final_response_text = None
        final_event = None
        all_events = []  # Store all events to find grounding metadata
        
        logger.info(f"Processing agent response for session: {session_id}")
        
        # Make sure we use the correct parameter names for run_async
        events_generator = runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content
        )
        
        async for event in events_generator:
            all_events.append(event)
            # Log tool calls
            if hasattr(event, 'actions') and event.actions and hasattr(event.actions, 'tool_calls') and event.actions.tool_calls:
                for tool_call in event.actions.tool_calls:
                    logger.info(f"Tool call: {tool_call.name} with params: {tool_call.parameters}")
            
            if event.is_final_response():
                final_event = event
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
        
        logger.info(f"Agent processing complete with {len(all_events)} events")
        return {
            "author": final_event.author if final_event else "unknown",
            "content": final_event.content if final_event else None,
            "type": type(final_event).__name__ if final_event else "unknown",
            "final_response": True,
            "final_response_text": final_response_text,
            "final_event": final_event,
            "all_events": all_events
        }
    except Exception as e:
        logger.error(f"Error in process_agent_response: {str(e)}")
        return {"final_response_text": f"Error processing response: {str(e)}"}

# Main function to run the agent
def run_agent_with_content(content):
    """Run the agent with user content"""
    session_service, artifact_service, agent = get_services_and_agent()
    session_id = st.session_state.session_id
    
   
    
    # Create runner with the agent, session service, and artifact service
    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name=APP_NAME,
        artifact_service=artifact_service
    )
    
    # Handle async operations with event loop
    loop = asyncio.new_event_loop()
    try:
        # First ensure session exists
        loop.run_until_complete(ensure_session_exists(session_service, session_id))
        
        # Then run the agent
        response = loop.run_until_complete(process_agent_response(runner, content, session_id))
        return response.get("final_response_text", "No response received")
    except Exception as e:
        logger.error(f"Error in run_agent_with_content: {str(e)}")
        return f"Error: {str(e)}"
    finally:
        loop.close()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# File uploader - modified to accept multiple files
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Replace the problematic section in your Streamlit app with this:

if uploaded_files:
    # Display the uploaded images
    for i, file in enumerate(uploaded_files):
        st.image(file, caption=f"Image {i+1}")
    
    # Add text input for the user's question
    user_text = st.text_input("Ask a question about these images:", value="Please analyze these images")
    
    # Add a button to submit
    if st.button("Analyze Images"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)
            for file in uploaded_files:
                st.image(file)
        
        # FIXED: Process ALL images in a SINGLE agent call
        with st.chat_message("assistant"):
            with st.spinner("Analyzing images..."):
                # Create parts for ALL images in one content object
                parts = [genai_types.Part(text=user_text)]
                
                # Add all images to the same content
                for i, file in enumerate(uploaded_files):
                    image_bytes = file.getvalue()
                    parts.append(genai_types.Part.from_bytes(data=image_bytes, mime_type=file.type))
                
                # Create single content with all images
                user_content = genai_types.Content(role="user", parts=parts)
                
                # Make SINGLE agent call with all images
                response = run_agent_with_content(user_content)
                st.write(response)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear the uploaded image to allow for new uploads
        st.session_state.uploaded_files = None
        st.rerun()

# Text input for questions
if not uploaded_files:  # Only show chat input when not uploading an image
    user_input = st.chat_input("Ask a question about the image...")
    if user_input:
        # Add to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Create content
        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_input)]
        )
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_agent_with_content(user_content)
                st.write(response)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun() 
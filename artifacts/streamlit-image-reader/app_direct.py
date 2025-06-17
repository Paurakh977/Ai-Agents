"""
Image Reader Agent Streamlit App (Direct Implementation)

This version implements the agent directly without importing from the image_reader_agent package.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import asyncio

# Import ADK components
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("🚫 GOOGLE_API_KEY not found. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Configure the page
st.set_page_config(
    page_title="Image Reader Agent (Direct)",
    page_icon="🖼️",
    layout="wide",
)

# Constants
APP_NAME = "image_reader_agent_direct"
USER_ID = "streamlit_user"
GEMINI_MODEL = "gemini-2.0-flash-exp"
IMAGE_DIR = "uploaded_images"

# Ensure image directory exists
def ensure_image_directory_exists():
    """Ensure that the images directory exists, creating it if necessary."""
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Created images directory at {IMAGE_DIR}")
    return IMAGE_DIR

# Ensure image directory exists at startup
ensure_image_directory_exists()

# Define the analyze_image tool
def analyze_image(
    tool_context: ToolContext,
    image_id: str = None,
) -> dict:
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
            tool_context.load_artifact(filename=image_to_analyze)
            
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

# Define the before_model_callback
async def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse:
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
            image_artifact = genai_types.Part(
                inline_data=genai_types.Blob(data=image_data, mime_type=mime_type)
            )
            
            # Save as an artifact
            try:
                # Use await to properly handle the coroutine
                artifact_version = await callback_context.save_artifact(
                    filename=image_name, artifact=image_artifact
                )
                print(f"[Image Callback] Saved image as artifact: {image_name} (version {artifact_version})")
                
                # Store the current image in state
                callback_context.state["current_image"] = image_name
                callback_context.state["current_image_version"] = artifact_version
            except Exception as e:
                print(f"[Image Callback] Error saving image as artifact: {str(e)}")
                
        except Exception as e:
            print(f"[Image Callback] Error saving image: {str(e)}")

    # Log the total number of images processed
    if image_count > 0:
        print(f"[Image Callback] Saved {image_count} images to {image_dir}")

    # Continue with normal execution
    return None

# Create the Image Reader Agent
image_reader_agent = Agent(
    name="image_reader_agent",
    description="A specialized agent that analyzes and describes images uploaded by users.",
    model=GEMINI_MODEL,
    before_model_callback=before_model_callback,
    tools=[analyze_image],
    instruction="""
    # 🖼️ Image Reader Agent

    You are an advanced AI image analysis agent that can analyze images uploaded by users.
    Your goal is to provide detailed, accurate descriptions of any image a user uploads.
    
    ## Your Capabilities

    - Receive images directly from users
    - Analyze image content in detail
    - Describe people, objects, scenes, text, and other elements in the image
    - Answer questions about the image content
    - Provide context and insights about what's shown in the image
    
    ## Process
    
    1. When a user uploads an image, it will be automatically detected and processed
    2. The image will be made available to you through your multimodal capabilities
    3. Always look carefully at the ENTIRE image before responding
    4. If the user asks about specific parts of the image, focus your analysis on those areas
    
    ## Guidelines
    
    - Be precise and detailed in your descriptions
    - If an image contains text, always include a transcription of the visible text
    - If you're uncertain about any element in the image, acknowledge your uncertainty
    - If a user asks about something not visible in the image, politely explain that you don't see it
    - If multiple images are uploaded, analyze each one in sequence
    - When analyzing charts or data visualizations, try to extract the key data points and trends
    - For diagrams or technical images, explain the components and their relationships
    
    ## Initial Interaction
    
    - When a user first connects, welcome them and explain that they can upload images for analysis
    - If the user has already uploaded an image, immediately begin your analysis
    - Always be helpful, accurate, and respectful in your responses
    
    Remember, your primary goal is to provide accurate and helpful analysis of the visual content shared by users.
    """,
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess-{os.urandom(8).hex()}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize ADK Runner
@st.cache_resource
def initialize_runner():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=image_reader_agent,  # Using our directly created agent
        session_service=session_service,
        app_name=APP_NAME,
    )
    return runner

runner = initialize_runner()

# Main page
st.title("🖼️ Image Reader Agent (Direct Implementation)")
st.markdown(
    """
    Upload an image, and I'll analyze it for you using Google's Agent Development Kit (ADK) and Gemini multimodal capabilities.
    """
)

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This app directly implements the Image Reader Agent without importing the existing package.
        
        Simply upload an image, and the agent will:
        - Detect objects
        - Recognize people, scenes, and text
        - Describe the image content
        - Answer questions about it
        """
    )
    
    st.divider()
    
    # Debug information
    st.write("Debug Info:")
    st.caption(f"App Name: {APP_NAME}")
    st.caption(f"User ID: {USER_ID}")
    st.caption(f"Session ID: {st.session_state.session_id}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
chat_container = st.container()

# Function to run ADK agent
async def run_agent(user_content):
    try:
        # Get session or create if it doesn't exist
        current_session = runner.session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=st.session_state.session_id
        )
        if not current_session:
            current_session = runner.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=st.session_state.session_id
            )
        
        # Run the agent
        events = runner.run_async(
            user_id=USER_ID,
            session_id=st.session_state.session_id,
            user_content=user_content,
        )
        
        # Process events and collect response
        response_text = ""
        async for event in events:
            if hasattr(event, "content"):
                for part in event.content.parts:
                    if hasattr(part, "text"):
                        response_text += part.text
        
        return response_text
    
    except Exception as e:
        return f"Error: {str(e)}"

# Function to make synchronous calls to the async function
def run_agent_sync(user_content):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_agent(user_content))
    finally:
        loop.close()

# Display chat history
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user" and "image_data" in message:
                # Display the image the user uploaded
                st.image(message["image_data"])
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])

# User input area
with st.container():
    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    # Text input for questions
    user_input = st.chat_input("Ask about your image or upload a new one...")
    
    # Process the user's input or uploaded image
    if uploaded_file or user_input:
        user_message = {"role": "user", "content": user_input or "Please analyze this image."}
        
        # If a new image is uploaded
        if uploaded_file:
            # Convert to bytes for ADK
            image_bytes = uploaded_file.getvalue()
            
            # Create PIL Image for display in Streamlit
            image = Image.open(BytesIO(image_bytes))
            
            # Store image in message for display
            user_message["image_data"] = image
            
            # Add to chat history
            st.session_state.chat_history.append(user_message)
            
            # Display the message immediately
            with st.chat_message("user"):
                st.image(image)
                st.markdown(user_message["content"])
            
            # Create ADK content for the agent
            content = genai_types.Part.from_bytes(
                data=image_bytes, 
                mime_type=uploaded_file.type
            )
            
            # Prepare the message with both text and image
            message_parts = [
                genai_types.Part.from_text(user_message["content"]),
                content
            ]
            
            # Create the ADK user message
            user_content = genai_types.Content(
                role="user", 
                parts=message_parts
            )
            
        # If it's just text
        elif user_input:
            # Add to chat history
            st.session_state.chat_history.append(user_message)
            
            # Display the message immediately
            with st.chat_message("user"):
                st.markdown(user_message["content"])
            
            # Create ADK user content with just text
            user_content = genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(user_input)]
            )
        
        # Show a thinking indicator
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Run the agent
                response_text = run_agent_sync(user_content)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                # Display response
                st.markdown(response_text) 
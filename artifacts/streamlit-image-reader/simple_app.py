"""
Simplified Image Reader Agent Streamlit App

A minimal implementation that connects Streamlit with Google's ADK Image Reader Agent.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
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

# Constants
APP_NAME = "image_reader_agent_simple"
USER_ID = "streamlit_user"
GEMINI_MODEL = "gemini-2.0-flash-exp"
IMAGE_DIR = "uploaded_images"

# Ensure image directory exists
def ensure_image_directory_exists():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Created images directory at {IMAGE_DIR}")
    return IMAGE_DIR

# Ensure directory exists at startup
ensure_image_directory_exists()

# Define the analyze_image tool
def analyze_image(
    tool_context: ToolContext,
    image_id: str = None,
) -> dict:
    """
    Analyze an image using the multimodal capabilities of Gemini.
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

        # Make the image available to the model
        try:
            tool_context.load_artifact(filename=image_to_analyze)
            return {
                "status": "success",
                "message": f"Image {image_to_analyze} loaded for analysis.",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error loading image: {str(e)}",
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
    Callback that processes images from user messages.
    """
    # Get the last user message parts
    last_user_message_parts = []
    if llm_request.contents and llm_request.contents[-1].role == "user":
        if llm_request.contents[-1].parts:
            last_user_message_parts = llm_request.contents[-1].parts

    # Process any image parts
    image_count = 0
    image_dir = ensure_image_directory_exists()

    for part in last_user_message_parts:
        # Check if it's an image
        if (hasattr(part, "inline_data") and part.inline_data and 
            hasattr(part.inline_data, "mime_type") and 
            part.inline_data.mime_type and 
            part.inline_data.mime_type.startswith("image/") and
            hasattr(part.inline_data, "data") and 
            part.inline_data.data):
            
            # Extract image data
            mime_type = part.inline_data.mime_type
            image_data = part.inline_data.data
            image_count += 1
            
            # Get file extension
            extension = mime_type.split("/")[-1]
            if extension == "jpeg":
                extension = "jpg"
            
            # Generate filename and save
            image_name = f"uploaded_image_{image_count}.{extension}"
            image_path = os.path.join(image_dir, image_name)
            
            try:
                # Save to disk
                with open(image_path, "wb") as f:
                    f.write(image_data)
                
                # Save as artifact
                image_artifact = genai_types.Part(
                    inline_data=genai_types.Blob(data=image_data, mime_type=mime_type)
                )
                
                artifact_version = await callback_context.save_artifact(
                    filename=image_name, artifact=image_artifact
                )
                
                # Store reference in state
                callback_context.state["current_image"] = image_name
                callback_context.state["current_image_version"] = artifact_version
                
            except Exception as e:
                print(f"Error saving image: {str(e)}")
    
    # Continue with normal processing
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
    - Analyze image content in detail
    - Describe people, objects, scenes, text, and other elements in the image
    - Answer questions about the image content
    
    ## Guidelines
    - Be precise and detailed in your descriptions
    - If an image contains text, include a transcription
    - When analyzing charts or visualizations, extract key data points
    """,
)

# Initialize ADK Runner
@st.cache_resource
def get_runner():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=image_reader_agent,
        session_service=session_service,
        app_name=APP_NAME,
    )
    return runner

# Process events and get response text
async def process_agent_response(events):
    response_text = ""
    async for event in events:
        if hasattr(event, "content") and event.content and hasattr(event.content, "parts"):
            for part in event.content.parts:
                if hasattr(part, "text"):
                    response_text += part.text
    return response_text

# Synchronous wrapper for async agent execution
def run_agent(content):
    runner = get_runner()
    
    # Create or get session
    session_id = st.session_state.get("session_id", f"sess-{os.urandom(4).hex()}")
    st.session_state["session_id"] = session_id
    
    try:
        # Run the agent
        events = runner.run_async(USER_ID, session_id, content)
        
        # Process the response
        loop = asyncio.new_event_loop()
        response = loop.run_until_complete(process_agent_response(events))
        loop.close()
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("🖼️ Simple Image Reader")
st.markdown("Upload an image for analysis with Google's ADK Agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded file
if uploaded_file:
    # Create message parts
    user_message = "Please analyze this image"
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_message)
    
    # Create content for the agent
    image_bytes = uploaded_file.getvalue()
    message_parts = [
        genai_types.Part.from_text(user_message),
        genai_types.Part.from_bytes(data=image_bytes, mime_type=uploaded_file.type)
    ]
    
    user_content = genai_types.Content(role="user", parts=message_parts)
    
    # Get response from agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing image..."):
            response = run_agent(user_content)
            st.write(response)
    
    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Chat input for follow-up questions
user_input = st.chat_input("Ask about the image...")
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Create content for the agent
    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(user_input)]
    )
    
    # Get response from agent
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = run_agent(user_content)
            st.write(response)
    
    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response}) 
"""
Image Reader Agent Streamlit App

This app allows users to upload images and have them analyzed using the ADK Image Reader Agent.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import asyncio

# Add the parent directory to the path to import the image_reader_agent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from image_reader_agent package
from .agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

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
    page_title="Image Reader Agent",
    page_icon="🖼️",
    layout="wide",
)

# Constants
APP_NAME = "image_reader_agent"
USER_ID = "streamlit_user"

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
        agent=root_agent,
        session_service=session_service,
        app_name=APP_NAME,
    )
    return runner

runner = initialize_runner()

# Main page
st.title("🖼️ Image Reader Agent")
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
        This app uses a Google ADK agent to analyze images.
        
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

# Async function to process agent responses
async def process_agent_response(events):
    """Process events from the agent runner and collect response text"""
    response_text = ""
    async for event in events:
        if hasattr(event, "content"):
            for part in event.content.parts:
                if hasattr(part, "text"):
                    response_text += part.text
    return response_text

# Synchronous wrapper for running the agent
def run_agent_sync(user_content):
    """Run the agent synchronously by managing an event loop for the async calls"""
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
        
        # Process events in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_text = loop.run_until_complete(process_agent_response(events))
            return response_text
        finally:
            loop.close()
            
    except Exception as e:
        return f"Error: {str(e)}"

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
                # Run the agent using our sync wrapper
                response_text = run_agent_sync(user_content)
                    
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                # Display response
                st.markdown(response_text) 
"""
Minimal Streamlit + ADK Image Reader App
"""

import os
import streamlit as st
from dotenv import load_dotenv
import asyncio
from typing import Dict, Optional, List, Union
import base64

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
def analyze_image(tool_context: ToolContext, image_id: Optional[str] = None) -> dict:
    """Tool to analyze an image with Gemini's multimodal capabilities"""
    try:
        # Get image to analyze (from parameter or state)
        if image_id:
            image_to_analyze = image_id
        elif "current_image" in tool_context.state:
            image_to_analyze = tool_context.state["current_image"]
        else:
            return {"error": "No image found to analyze"}
            
        # Load the artifact for the model
        tool_context.load_artifact(filename=image_to_analyze)
        return {"success": f"Image {image_to_analyze} loaded for analysis"}
        
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

# Define callback to process uploaded images
async def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """Process images in user messages"""
    
    # Extract user message parts
    if not llm_request.contents or llm_request.contents[-1].role != "user":
        return None
        
    user_parts = llm_request.contents[-1].parts if llm_request.contents[-1].parts else []
    
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
        mime_type = part.inline_data.mime_type
        extension = mime_type.split("/")[-1]
        if extension == "jpeg":
            extension = "jpg"
            
        # Save image with sequence number
        image_name = f"uploaded_image_{i+1}.{extension}"
        
        # Save as artifact
        image_artifact = genai_types.Part(
            inline_data=genai_types.Blob(
                data=image_data,
                mime_type=mime_type
            )
        )
        
        # Save artifact and update state
        artifact_version = await callback_context.save_artifact(
            filename=image_name, 
            artifact=image_artifact
        )
        
        callback_context.state["current_image"] = image_name
        callback_context.state["current_image_version"] = artifact_version
        
    return None

# Create agent
def create_agent():
    """Create the image reader agent"""
    return Agent(
        name="image_reader_agent",
        description="Analyzes images uploaded by users",
        model=GEMINI_MODEL,
        before_model_callback=before_model_callback,
        tools=[analyze_image],
        instruction="""
        # Image Analysis Agent
        
        You analyze images that users upload and provide detailed descriptions.
        
        When a user uploads an image:
        1. Describe what you see in detail
        2. If text is visible, transcribe it
        3. Answer any questions about the image content
        
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
        st.error(f"Session error: {str(e)}")
        return False

# Process agent response
async def process_agent_response(runner, content, session_id):
    """Run the agent and process the response"""
    try:
        final_response_text = None
        final_event = None
        all_events = []  # Store all events to find grounding metadata
        
        print(content)
        
        # Make sure we use the correct parameter names for run_async
        events_generator = runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content
        )
        
        async for event in events_generator:
            all_events.append(event)
            
            if event.is_final_response():
                final_event = event
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
        
        print(all_events)
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
        artifact_service=artifact_service  # Add artifact service to the Runner
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
        return f"Error: {str(e)}"
    finally:
        loop.close()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# File uploader - modified to accept multiple files
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

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
        
        # Process each image and collect responses
        with st.chat_message("assistant"):
            with st.spinner("Analyzing images..."):
                responses = []
                
                for i, file in enumerate(uploaded_files):
                    # Get file bytes
                    image_bytes = file.getvalue()
                    
                    # Create content for agent
                    parts = [
                        genai_types.Part(text=f"{user_text} (Image {i+1} of {len(uploaded_files)})"),
                        genai_types.Part.from_bytes(data=image_bytes, mime_type=file.type)
                    ]
                    user_content = genai_types.Content(role="user", parts=parts)
                    
                    # Get response
                    response = run_agent_with_content(user_content)
                    responses.append(f"**Analysis of Image {i+1}**:\n{response}")
                
                # Combine responses
                combined_response = "\n\n".join(responses)
                st.write(combined_response)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": combined_response})
        
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
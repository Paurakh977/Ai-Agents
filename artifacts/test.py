import streamlit as st
import os
from datetime import datetime
from PIL import Image

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Page configuration
st.set_page_config(page_title="Chat Interface", page_icon="💬", layout="centered")

st.title("💬 Chat Interface")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display text message
        if message["content"]:
            st.write(message["content"])
        
        # Display uploaded files
        if "files" in message and message["files"]:
            st.write("📎 **Attached files:**")
            cols = st.columns(min(len(message["files"]), 4))
            for i, file_info in enumerate(message["files"]):
                with cols[i % 4]:
                    try:
                        img = Image.open(file_info["path"])
                        st.image(img, caption=file_info["name"], width=150)
                    except:
                        st.write(f"📄 {file_info['name']}")

# Input section at the bottom
st.markdown("---")

# File uploader
uploaded_files = st.file_uploader(
    "📎 Attach images",
    accept_multiple_files=True,
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
    key="file_uploader",
    label_visibility="collapsed"
)

# Preview currently selected images
if uploaded_files:
    st.write("**Selected images:**")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, file in enumerate(uploaded_files):
        with cols[i % 4]:
            img = Image.open(file)
            st.image(img, caption=file.name, width=120)

# Text input and send button
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("Message...", key="user_input", placeholder="Type a message...")
with col2:
    send_button = st.button("Send", type="primary")

# Process message when send button is clicked
if send_button and (user_input or uploaded_files):
    # Prepare message data
    message_data = {
        "role": "user",
        "content": user_input if user_input else "",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "files": []
    }
    
    # Handle file uploads
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{uploaded_file.name}"
            file_path = os.path.join(UPLOAD_DIR, filename)
            
            # Save file to uploads directory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Add file info to message
            file_info = {
                "name": uploaded_file.name,
                "path": file_path,
                "size": uploaded_file.size,
                "type": uploaded_file.type
            }
            message_data["files"].append(file_info)
    
    # Add message to chat history
    st.session_state.messages.append(message_data)
    
    # Create assistant response
    assistant_response = "Message received"
    if message_data["content"]:
        assistant_response += f": \"{message_data['content']}\""
    
    if message_data["files"]:
        file_names = [f["name"] for f in message_data["files"]]
        assistant_response += f"\n📎 Files uploaded: {', '.join(file_names)}"
    
    assistant_message = {
        "role": "assistant",
        "content": assistant_response,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.session_state.messages.append(assistant_message)
    
    # Clear input (rerun will handle this)
    st.rerun()
# 🖼️ Image Reader Agent - Streamlit App

This Streamlit app integrates with the Image Reader Agent from Google's Agent Development Kit (ADK) to analyze images using Gemini multimodal capabilities.

## Features

- Upload images via the Streamlit interface
- Get detailed analysis of image content
- Ask follow-up questions about the image
- Interactive chat interface
- Multimodal analysis using Google Gemini

## Setup

### Prerequisites

- Python 3.10+
- A Google API key with access to Gemini models

### Installation

1. Clone the repository and navigate to this directory:

```bash
git clone <repository-url>
cd streamlit-image-reader
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Google API key:

```bash
# Create .env file from example
cp .env.example .env

# Edit the .env file and add your API key
# GOOGLE_API_KEY=your-actual-api-key-here
```

### Running the App

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How to Use

1. Upload an image using the file uploader
2. The agent will automatically analyze the image
3. Ask follow-up questions about the image
4. Upload new images as needed for different analyses

## Implementation Details

This app connects to the existing image-reader-agent using ADK's Runner and SessionService, and integrates it with a Streamlit interface for easy user interaction.

The application:
- Initializes an ADK Runner with the image reader agent
- Creates a user-friendly interface with Streamlit
- Converts uploaded images into the format required by ADK
- Displays agent responses in a chat interface

## Troubleshooting

- If you see API key errors, ensure your `.env` file contains a valid `GOOGLE_API_KEY`
- For issues with image analysis, check that the image format is supported (PNG, JPG, JPEG)
- If the app crashes, check the console for error messages 
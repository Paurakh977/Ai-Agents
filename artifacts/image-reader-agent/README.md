# Image Reader Agent

A simple AI agent built with Google's Agent Development Kit (ADK) that can receive and analyze images.

## Features

- Upload images directly to the agent for analysis
- Get detailed descriptions of image content
- Ask questions about specific elements in the image
- Uses Gemini 2.0 Flash model for multimodal understanding

## Installation

1. Clone this repository
```bash
git clone <repository-url>
cd image-reader-agent
```

2. Set up a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure your Google API key in a `.env` file:
```
GOOGLE_API_KEY=your_google_api_key
```

## Usage

Run the agent with:
```bash
adk run .
```

For a web interface, run:
```bash
adk web
```

Then select your agent in the web UI and upload images for analysis.

## Project Structure

```
image-reader-agent/
├── image_reader_agent/
│   ├── __init__.py
│   ├── agent.py        # Main agent definition
│   ├── callbacks.py    # Image upload handling
│   ├── constants.py    # Configuration constants
│   ├── utils.py        # Utility functions
│   └── tools/
│       ├── __init__.py
│       └── analyze_image.py  # Image analysis tool
├── images/             # Directory for storing uploaded images
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10+
- Google ADK
- Google Generative AI Python SDK 
# How to Run the Image Reader App

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key**
   Create a `.env` file in the same directory with your Google API key:
   ```
   GOOGLE_API_KEY=your-api-key-here
   ```
   Get your API key from: https://aistudio.google.com/app/apikey

## Running the App

1. **Start the Streamlit App**
   ```bash
   streamlit run minimal_app.py
   ```
   
2. **Using the App**
   - Upload an image using the file uploader
   - The agent will automatically analyze the image
   - Ask follow-up questions using the chat input

## Troubleshooting

- If you get API key errors, check that your `.env` file contains the correct key
- Make sure you have a working internet connection
- If the app crashes, check the console output for errors 
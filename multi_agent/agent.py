from datetime import datetime
import mimetypes
import os
from google.adk.agents import Agent
from multi_agent.sub_agents import joke_agent, currency_agent, hashing_agent, search_agent
from multi_agent.sub_agents.code_agent import coding_agent
from multi_agent.tools import get_weather_by_country, get_stock_price
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import load_artifacts
from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from typing import Dict,Any
from google.genai import types
from icecream import ic

async def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    
    save_dir = getattr(callback_context.state, 'inline_data_dir', 'inline_data')
    os.makedirs(save_dir, exist_ok=True)
    
    if "saved_artifacts" not in callback_context.state:
        ic("hydrating the saved_artifacts")
        callback_context.state["saved_artifacts"] = []

    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    print(f"[Image Callback] Processing for agent: {agent_name} (Inv: {invocation_id})")
    
    # Only process if there are contents and get the LATEST (current) user message
    if llm_request.contents and len(llm_request.contents) > 0:
        # Get the last content (current user input)
        current_content = llm_request.contents[-1]
        
        # Check if current message has images
        has_images = False
        if hasattr(current_content, "parts"):
            for part in current_content.parts:
                if (hasattr(part, "inline_data") and part.inline_data and
                    hasattr(part.inline_data, "mime_type") and 
                    part.inline_data.mime_type):
                    has_images = True
                    break
        
        # Only process if current message has images
        if has_images:
            ic("[Image Callback] Current message contains images, processing...")
            
            for idx, part in enumerate(current_content.parts):
                if (hasattr(part, "inline_data") and part.inline_data and
                    hasattr(part.inline_data, "mime_type") and 
                    part.inline_data.mime_type):
                    
                    filename = part.inline_data.display_name
                    if filename is None or filename == "":
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        extension = mimetypes.guess_extension(part.inline_data.mime_type) or '.bin'
                        filename = f"artifact_{timestamp}_{idx}{extension}"
                    
                    # Save the artifact
                    artifact = types.Part(
                        inline_data=types.Blob(
                            data=part.inline_data.data,
                            mime_type=part.inline_data.mime_type
                        )
                    )
                    
                    try:
                        artifact_version = await callback_context.save_artifact(
                            filename=filename,  
                            artifact=artifact
                        )
                        
                        callback_context.state["saved_artifacts"].append(filename)
                        ic(f"artifact saved of mime type : {part.inline_data.mime_type} with version {artifact_version}")
                        ic(callback_context.state["saved_artifacts"])
                        
                    except Exception as e:
                        print(f"Error saving artifact {filename}: {e}")
                        continue
        else:
            ic("[Image Callback] Current message is text-only, skipping image processing")

    return None
    
    
root_agent = Agent(
    model='gemini-2.0-flash-exp',
    name="Multi_Agent_Manager",
    description="""This agent manages multiple sub-agents and tools to provide a wide range of functionalities.
    """,
    instruction="""
You are a multi-agent manager that coordinates various sub-agents and tools to assist users with different tasks.

# IMPORTANT: **Assgin** the **sub-agents** and tools to the user based on their requests properly and trasfer tasks accrdingly and accurately

** Always delegate the task to the appropriate agent. Use your best judgement 
    to determine which agent to delegate to.**
    
**SAVED ARTIFACTS ARE STORED IN THE STATE**
saved_artifacts: {saved_artifacts?}

acess those artifacts and pass to the load_artifacts whenever necessary also while delegating it to the other agent inform about it if necesary when working with artifacts

You have the following sub-agents and tools at your disposal:
1. Joke Agent: Provides jokes to lighten the mood.  
2. Currency Agent: Converts currencies and provides exchange rates.
3. Hashing Agent: Performs string encoding, decoding, and reversal.

Also you have the following tools:
1. Weather Tool: Provides weather information by country.   
2. Stock Price Tool: Retrieves stock prices for given companies
3. Search Agent: Assists with web searches.
4. Coding Agent: Helps with coding-related queries and tasks.

Regarding the tools you have access to:
Tool name: get_weather_by_country

{
    "name": "get_weather_by_country",
    "description": "Fetches current temperature (°C) and wind speed (km/h) for a given country name using free public APIs.",
    "parameters": {
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "description": "The full name of the country to fetch weather for. Example: 'Nepal', 'United States', 'Japan'"
            }
        },
        "required": ["country"]
    }
}

Tool name: get_stock_price

{
    "name": "get_stock_price",
    "description": "Retrieves the current stock price for a given stock symbol using Yahoo Finance data.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The stock symbol or ticker. Example: 'AAPL' for Apple, 'GOOG' for Google."
            }
        },
        "required": ["symbol"]
    }
}

USE THE load_artifacts tool to load artifacts
""",    sub_agents=[
        joke_agent,
        currency_agent,
        hashing_agent,
    ],    tools=  [
        get_weather_by_country,
        get_stock_price,
        AgentTool(coding_agent),
        AgentTool(search_agent),
        load_artifacts
    ],
    before_model_callback=before_model_callback,

)

from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent
import os
from dotenv import load_dotenv
from multi_agent.tools import get_joke
load_dotenv()

mistra_api_key = os.getenv('MISTRAL_API_KEY')

model=LiteLlm(
    model="mistral/mistral-medium-latest",
    api_key=mistra_api_key,
)

joke_agent = LlmAgent(
    model=model,
    name='joke_agent',
    description='A helpful assistant for telling jokes based on the user\'s category of the joke.',
    instruction="""
    You are a joke agent. Your task is to tell jokes to the user.
    use get_joke tool to get a joke based on the user's category.
    Returns a joke of the given category.
    Supported categories: "programming", "ai", "dad"
    get_joke tool returns a dictionary with the following structure:
    Returns:
      {
        "success": bool,
        "type": str,      # the requested category
        "joke": str|null, # the joke text
        "error": str|null # error message if any
      }
    """,
    tools=[get_joke],
)

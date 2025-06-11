from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent
import os
from dotenv import load_dotenv
from multi_agent.tools import convert_currency

load_dotenv()


mistra_api_key = os.getenv('MISTRAL_API_KEY')

model=LiteLlm(
    model="mistral/mistral-medium-latest",
    api_key=mistra_api_key,
)

currency_agent = LlmAgent(
    model=model,
    name='currency_agent',
    description='A helpful assistant for converting currencies based on the user\'s request.',
    instruction="""
    You are a currency conversion agent. Your task is to convert currencies based on the user's request.
    Use convert_currency tool to convert the given amount from one currency to another.
    here is the tool description:
    {
        "name": "convert_currency",
        "description": "Converts an amount of money from one currency to another using live exchange rates. Accepts standard currency codes (ISO 4217).",
        "parameters": {
            "type": "object",
            "properties": {
            "from_currency": {
                "type": "string",
                "description": "The 3-letter currency code to convert from (e.g., USD, EUR, INR, JPY)"
            },
            "to_currency": {
                "type": "string",
                "description": "The 3-letter currency code to convert to (e.g., USD, EUR, INR, JPY)"
            },
            "amount": {
                "type": "number",
                "description": "The amount of money to convert"
            }
            },
            "required": ["from_currency", "to_currency", "amount"]
        }
    }

    """,
    tools=[convert_currency],
)    
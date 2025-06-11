from google.adk.agents import Agent
from multi_agent.sub_agents import joke_agent, currency_agent, hashing_agent, search_agent
from multi_agent.sub_agents.code_agent import coding_agent
from multi_agent.tools import get_weather_by_country, get_stock_price
from google.adk.tools.agent_tool import AgentTool

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

""",    sub_agents=[
        joke_agent,
        currency_agent,
        hashing_agent,
    ],    tools=  [
        get_weather_by_country,
        get_stock_price,
        AgentTool(coding_agent),
        AgentTool(search_agent),
    ]
)

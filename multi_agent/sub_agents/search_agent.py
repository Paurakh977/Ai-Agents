from google.adk.agents import Agent
from google.adk.tools import google_search

search_agent = Agent(
    model='gemini-2.0-flash-exp',
    name='search_agent',
    description='A helpful assistant for searching the web.',
    instruction='Search the web to find information that can help answer user questions.',
    tools=[google_search],
)
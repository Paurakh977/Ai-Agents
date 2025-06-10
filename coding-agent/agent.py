from google.adk.agents import Agent
from google.adk.code_executors import BuiltInCodeExecutor
root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='coding_assistant_agent',
    code_executor=BuiltInCodeExecutor(),
    description='A coding assistant that helps with programming tasks',
    instruction='You are a coding assistant. Help the user with their programming tasks by providing code snippets, explanations, and debugging assistance. Be concise and clear in your responses.',
)

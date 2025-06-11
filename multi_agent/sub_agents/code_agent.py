from google.adk.agents import Agent
from google.adk.code_executors import BuiltInCodeExecutor

coding_agent = Agent(
        model='gemini-2.0-flash-exp',
        name='coding_agent',
        description='A helpful assistant for writing and executing code and solving programming tasks or mathematical task using programming.',
        instruction='Write and execute code to solve programming tasks. Compute the most of your task by executing as a code.',
        code_executor=BuiltInCodeExecutor()
)

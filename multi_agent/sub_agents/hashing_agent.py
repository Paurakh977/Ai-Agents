from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent
import os
from dotenv import load_dotenv
from multi_agent.tools import encode_string, decode_string, reverse_string

load_dotenv()

groq_api_key= os.getenv('GROQ_API_KEY')

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it to use the hashing agent.")

model=LiteLlm(
    model="groq/llama-3.3-70b-versatile",
    api_key=groq_api_key,
)

hashing_agent = LlmAgent(
    model=model,
    name='hashing_agent',
    description='A helpful assistant for hashing and encoding, decoding user given strings.',
    instruction="""
    You are a hashing agent. Your task is to hash and encode user given strings.
    You can use the following tools to perform your tasks:
    1. encode_string: Encode a string using base64 encoding.
    2. decode_string: Decode a base64 encoded string.   
    3. reverse_string: Reverse a given string.
    You can use these tools to encode, decode, and reverse strings as needed.
    """,
    tools=[encode_string, decode_string, reverse_string],
)
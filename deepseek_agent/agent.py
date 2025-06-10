from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
import os       
from dotenv import load_dotenv
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

groq_model= LiteLlm(
    model="groq/llama-3.3-70b-versatile",
    api_key=groq_api_key,
)

openrouter_model= LiteLlm(
    model="openrouter/deepseek/deepseek-r1-0528:free",
    api_key=openrouter_api_key,
)

root_agent = LlmAgent(
    name="OpenRouter_DeepSeek_Agent",
    model=openrouter_model,
    description="A DeepSeek agent that can answer questions and perform tasks.",
    instruction="You are a DeepSeek agent. Help the user with their programming tasks by providing code snippets, explanations, and debugging assistance. Be concise and clear in your responses.",
)

# root_agent = LlmAgent(
#     name="groq_llama_agent",
#     model=groq_model,
#     description="A Groq Llama agent that can answer questions and perform tasks.",
#     instruction="You are a Groq Llama agent. Help the user with their programming tasks by providing code snippets, explanations, and debugging assistance. Be concise and clear in your responses.",
# )

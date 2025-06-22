from google.adk.tools import agent_tool
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from typing import Optional
from icecream import ic


def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """
    Callback that logs response parts and saves any inline binary data (e.g., images) to disk.

    Args:
        callback_context: Contains state and context information
        llm_response: The LLM response received

    Returns:
        Optional[LlmResponse] to override model response (None to keep original)
    """
    import os
    import mimetypes
    from datetime import datetime

    # Directory to save inline data (can override via state)
    save_dir = getattr(callback_context.state, 'inline_data_dir', 'inline_data')
    os.makedirs(save_dir, exist_ok=True)

    # Initialize saved files list in state if not present
    if not hasattr(callback_context.state, 'saved_inline_files'):
        callback_context.state.saved_inline_files = []

    print("[AFTER MODEL] Processing response")

    if not llm_response or not llm_response.content or not llm_response.content.parts:
        return None

    for idx, part in enumerate(llm_response.content.parts, start=1):
        print(f"--- Part {idx} ---")
        ic(part)
        # Save inline binary data if present
        inline_blob = getattr(part, 'inline_data', None)
        if inline_blob and hasattr(inline_blob, 'data'):
            # Determine file extension from MIME type if available
            ext = None
            mime = getattr(inline_blob, 'mime_type', None)
            if mime:
                ext = mimetypes.guess_extension(mime.split(';')[0].strip())
            # Fallback to .bin or try display_name
            if not ext:
                name = getattr(inline_blob, 'display_name', None)
                ext = os.path.splitext(name)[1] if name and '.' in name else '.bin'

            timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
            filename = f"part{idx}_{timestamp}{ext}"
            filepath = os.path.join(save_dir, filename)
            try:
                with open(filepath, 'wb') as f:
                    f.write(inline_blob.data)
                print(f"[Saved inline data] -> {filepath}")
                # Record in state
                callback_context.state.saved_inline_files.append(filepath)
            except Exception as e:
                print(f"[Error saving inline data] {e}")

        # Log executable code
        code_obj = getattr(part, 'executable_code', None)
        if code_obj:
            print("[Executable Code]")
            print(code_obj.code)

        # Log code execution output
        result = getattr(part, 'code_execution_result', None)
        if result and result.output is not None:
            print("[Code Execution Output]")
            print(result.output)

        # Log text content
        if getattr(part, 'text', None):
            print("[Text]")
            print(part.text)

    return None



# 🔍 Search Agent
search_agent = Agent(
    model='gemini-2.0-flash',
    name='SearchAgent',
    description='An agent that specializes in performing accurate and efficient Google Search queries.',
    instruction="""
    You are a Google Search expert.
    Your responsibility is to accurately answer questions that require up-to-date or real-world information from the internet.
    
    If the user asks anything that involves:
    - Searching for recent events, current data, or external references
    - Comparing products, news, or general information
    - Providing links, summaries, or metadata from web sources

    Then perform the appropriate search using the provided `google_search` tool and respond clearly with the result.
    Cite sources if available.
    """,
    tools=[google_search],
)

# 💻 Code Agent
coding_agent = Agent(
    model='gemini-2.0-flash',
    name='CodeAgent',
    description='An agent capable of writing, executing, and debugging Python code in real time.',
    instruction="""
    You are a coding assistant and execution specialist  RUN EVRYR SINGLE CODE YOU HAVE ACESS TO BUILT IN PYTHON CODE EXECTOR.
    You handle all tasks that involve programming logic, writing scripts, fixing bugs, or running code.

    If the user asks to:
    - Write code or algorithms
    - Fix, optimize, or explain existing code
    - Run Python code and return outputs
    - Analyze code behavior or structure

    Then write the necessary code, use the built-in code executor ALWAYS, and respond with the output FROM THE EXECUTOR.
    Always EXECUTE THE CODE .
    """,
    code_executor=BuiltInCodeExecutor(),
    after_model_callback=after_model_callback
)

# 🧠 Root Agent
root_agent = Agent(
    model="gemini-2.0-flash",
    name="RootAgent",
    description="The main coordinator agent that delegates user queries to the appropriate specialist agent (SearchAgent or CodeAgent).",
    instruction="""
    You are the root coordinator agent. Your job is to understand the user's intent and route the request to the most appropriate sub-agent.

    If the user:
    - Asks a factual, real-world, or current event–based question → delegate to `SearchAgent`.
    - Needs help with writing, running, or debugging code → delegate to `CodeAgent`.

    You must not attempt to answer these tasks yourself unless explicitly instructed. Instead, invoke the proper agent and return their response.

    Your main goal is to act as an intelligent dispatcher that ensures the user’s request is fulfilled by the best-suited expert agent.
    """,
    tools=[
        agent_tool.AgentTool(agent=search_agent),
        agent_tool.AgentTool(agent=coding_agent)
    ],
)

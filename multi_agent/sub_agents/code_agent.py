from google.adk.agents import Agent
from google.adk.code_executors import BuiltInCodeExecutor,VertexAiCodeExecutor
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from typing import Optional 
import os
import mimetypes
from datetime import datetime
from icecream import ic
from google.genai import types

async def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    available_artifacts=callback_context.state["saved_artifacts"]
    ic(available_artifacts)
    try:
        for artifact in available_artifacts:
            result = await callback_context.load_artifact(artifact)
            ic(result.inline_data.display_name, result.inline_data.mime_type)
        ic("loaded artifacts sucesfully")
    except Exception as e:
        ic(e)

async def after_model_callback(
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
                
                # save it as artifact
                artifact=types.Part(
                    inline_data=types.Blob(
                        data=part.inline_data.data,
                        mime_type=part.inline_data.mime_type
                    )
                )
                
                artifact_version = await callback_context.save_artifact(
                    filename=filename,  
                    artifact=artifact
                )
                
                # Record in state
                callback_context.state.saved_inline_files.append(filepath)
                callback_context.state["statesaved_artifacts"].append(filename)

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

coding_agent = Agent(
        model='gemini-2.0-flash-exp',
        name='coding_agent',
        description='A helpful assistant for writing and executing code and solving programming tasks or mathematical task using programming.',
        instruction='Write and execute code to solve programming tasks. Compute the most of your task by executing as a code. Always EXECUTE CODE RUN THE CODE USING YOUR BUILT IN CODE EXECUTOR AND SHOW THE OUTPUT TO THE USER NO MATTER WHAT. THIS IS MANDATORY. RESPOND WITH EXECUTABLE CODE NOT WITH JUST TEXT.  **SAVED ARTIFACTS ARE STORED IN THE STATE  **saved_artifacts: {saved_artifacts?}',
        code_executor=BuiltInCodeExecutor(),
        after_model_callback=after_model_callback,
        before_model_callback=before_model_callback
)

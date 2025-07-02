from google.adk.agents import Agent
from google.adk.code_executors import BuiltInCodeExecutor
from datetime import datetime
import mimetypes
import os
from google.adk.agents import Agent
from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from icecream import ic

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

async def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    
    save_dir = getattr(callback_context.state, 'inline_data_dir', 'inline_data')
    os.makedirs(save_dir, exist_ok=True)
    
    if "saved_artifacts" not in callback_context.state:
        ic("hydrating the saved_artifacts")
        callback_context.state["saved_artifacts"] = []

    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    print(f"[Image Callback] Processing for agent: {agent_name} (Inv: {invocation_id})")
    
    # Only process if there are contents and get the LATEST (current) user message
    if llm_request.contents and len(llm_request.contents) > 0:
        # Get the last content (current user input)
        current_content = llm_request.contents[-1]
        
        # Check if current message has images
        has_images = False
        if hasattr(current_content, "parts"):
            for part in current_content.parts:
                if (hasattr(part, "inline_data") and part.inline_data and
                    hasattr(part.inline_data, "mime_type") and 
                    part.inline_data.mime_type):
                    has_images = True
                    break
        
        # Only process if current message has images
        if has_images:
            ic("[Image Callback] Current message contains images, processing...")
            
            for idx, part in enumerate(current_content.parts):
                if (hasattr(part, "inline_data") and part.inline_data and
                    hasattr(part.inline_data, "mime_type") and 
                    part.inline_data.mime_type):
                    
                    filename = part.inline_data.display_name
                    if filename is None or filename == "":
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        extension = mimetypes.guess_extension(part.inline_data.mime_type) or '.bin'
                        filename = f"artifact_{timestamp}_{idx}{extension}"
                    
                    # Save the artifact
                    artifact = types.Part(
                        inline_data=types.Blob(
                            data=part.inline_data.data,
                            mime_type=part.inline_data.mime_type
                        )
                    )
                    
                    try:
                        artifact_version = await callback_context.save_artifact(
                            filename=filename,  
                            artifact=artifact
                        )
                        
                        callback_context.state["saved_artifacts"].append(filename)
                        ic(f"artifact saved of mime type : {part.inline_data.mime_type} with version {artifact_version}")
                        ic(callback_context.state["saved_artifacts"])
                        
                    except Exception as e:
                        print(f"Error saving artifact {filename}: {e}")
                        continue
        else:
            ic("[Image Callback] Current message is text-only, skipping image processing")

    return None

    
root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='coding_assistant_agent',
    code_executor=BuiltInCodeExecutor(),
    description='A coding assistant that helps with programming tasks',
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    instruction='You are a coding assistant. Help the user with their programming tasks by providing code snippets, explanations, and debugging assistance. Be concise and clear in your responses.',
)

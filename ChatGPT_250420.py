import json
import os
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import docx
import pandas as pd
from time import sleep
import tiktoken
import logging
from typing import List, Dict, Tuple, Optional, Any # Added Any for message content type
import io
import base64 # Import base64 for image encoding

# ... (Logging, Constants, Page Setup, Client Initialization remain mostly the same) ...

# ------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------
MODEL_CONTEXT_LIMITS = {
    "gpt-3.5-turbo": 16385, # Primarily text
    "gpt-4": 8192, # Primarily text
    "gpt-4-32k": 32768, # Primarily text
    "gpt-4o": 128000, # Multimodal (Text, Vision, Audio)
    # Add other relevant multimodal models if needed, e.g., "gpt-4-turbo"
    "gpt-4-turbo": 128000, # Example name for GPT-4 Turbo, context might vary
}
# Define a list of models that support vision/multimodal input
MULTIMODAL_VISION_MODELS = ["gpt-4o", "gpt-4-turbo"] # Add "gpt-4-vision-preview" if needed

DEFAULT_ENCODING = "cl100k_base"
CHUNK_SIZE = 2000
RESERVED_TOKENS = 1500
HISTORY_FILE = "history/chat_history.json"
CHUNK_PROMPT_FOR_SUMMARY = 'Summarize the key points of this text chunk in 2-3 concise bullet points, focusing on the main information.'

# ... (Helper functions: get_tokenizer, num_tokens_from_string remain the same) ...

# num_tokens_from_messages needs update for multimodal content (array of parts)
# However, correctly calculating tokens for image content manually is complex.
# We will keep the text-only token calculation as is for now and rely on
# the API to handle multimodal token counting and catch potential context errors.
def num_tokens_from_messages(messages: List[Dict[str, Any]], encoding: tiktoken.Encoding) -> int:
    """Calculates tokens for text parts in messages (simplified for multimodal)."""
    num_tokens = 0
    for message in messages:
        num_tokens += 4 # base tokens per message

        # Handle content which can be a string or a list of parts
        content = message.get("content")
        if isinstance(content, str):
             num_tokens += num_tokens_from_string(content, encoding)
        elif isinstance(content, list):
             for part in content:
                 if part.get("type") == "text" and "text" in part:
                      num_tokens += num_tokens_from_string(part["text"], encoding)
                 # Note: This does NOT calculate tokens for images ("image_url")
                 # Image token calculation is resolution-dependent and complex.
                 # We proceed without precise image token count for context fitting.

        if "name" in message:
            num_tokens -= 1 # name token

    num_tokens += 2 # assistant reply start tokens
    return num_tokens

# read_file remains the same, as it converts to text or returns error for images

@st.cache_data(show_spinner=False)
def summarize_document(text: str, filename: str, model: str, tokenizer: tiktoken.Encoding) -> Tuple[str, Optional[str]]:
   # ... (summarize_document remains the same) ...
   pass # Placeholder

# load_history and save_history remain the same, they handle text messages

# ------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------
if 'messages' not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = load_history(HISTORY_FILE) # message content can be List[Dict]
    logging.info(f"Session initialized. Loaded {len(st.session_state.messages)} messages.")

if 'doc_summaries' not in st.session_state:
    st.session_state.doc_summaries: Dict[str, str] = {}
if 'processed_file_ids' not in st.session_state:
    st.session_state.processed_file_ids: set = set()

# File processing queues/states remain the same
if 'file_to_summarize' not in st.session_state:
    st.session_state.file_to_summarize: Optional[Dict] = None
if 'file_info_to_process_safely_captured' not in st.session_state:
     st.session_state.file_info_to_process_safely_captured: Optional[Dict] = None

# New state for holding image data to be sent with the *next* text prompt
if 'uploaded_image_for_next_prompt' not in st.session_state:
    st.session_state.uploaded_image_for_next_prompt: Optional[Dict] = None # Stores {'type': ..., 'bytes': ...}

# ------------------------------------------------------------------
# SIDEBAR: MODEL, MODE SELECTION & OPTIONS
# ------------------------------------------------------------------
st.sidebar.title("⚙️ 설정")

# Filter model selection to only include multimodal models
MODEL = st.sidebar.selectbox(
    '모델 선택 (멀티모달 지원)',
    MULTIMODAL_VISION_MODELS, # Use the filtered list
    index=MULTIMODAL_VISION_MODELS.index("gpt-4o") if "gpt-4o" in MULTIMODAL_VISION_MODELS else 0 # Default to gpt-4o if available
)
# Note: MAX_CONTEXT_TOKENS calculation still uses the broader list,
# but this is less critical as we are limiting selectable models.
MAX_CONTEXT_TOKENS = MODEL_CONTEXT_LIMITS.get(MODEL, 4096)


MODE = st.sidebar.radio('응답 모드', ('Poetic', 'Logical'), index=0, key='mode_selection')

st.sidebar.markdown("---")
st.sidebar.subheader("관리")

# build_full_session_content needs update to handle list content in messages if saving full history
# However, current save_history excludes system messages which might contain list content (summaries).
# Let's keep build_full_session_content as is, as it formats messages back to text for download.

# Download Button remains the same

# Clear Button remains the same

# ------------------------------------------------------------------
# SYSTEM PROMPT DEFINITION
# ------------------------------------------------------------------
# SYSTEM_PROMPT_CONTENT definition remains the same
SYSTEM_PROMPT_CONTENT = (
    'You are Liel, a poetic, emotionally intelligent chatbot with lyrical grace. Respond with warmth, creativity, and empathy. Use rich language and metaphors when appropriate.'
    if MODE == 'Poetic' else
    'You are Liel, a highly analytical assistant focused on logic and precision. Provide clear, structured, and concise answers. Use bullet points or numbered lists for clarity when needed.'
)
SYSTEM_PROMPT = {'role': 'system', 'content': SYSTEM_PROMPT_CONTENT}

# System prompt management in messages state remains the same

# ------------------------------------------------------------------
# UI HEADER
# ------------------------------------------------------------------
# UI Header remains the same

# ------------------------------------------------------------------
# FILE UPLOAD UI
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    '파일 업로드 (txt, pdf, docx, xlsx, jpg, png)',
    type=['txt', 'pdf', 'docx', 'xlsx', 'jpg', 'png'],
    key="file_uploader",
    help="텍스트, PDF, 워드, 엑셀 파일은 요약하여 컨텍스트에 포함하고, 이미지 파일은 다음 질문과 함께 모델에게 전송합니다." # Help text updated
)

# --- File Upload Handling and Queuing: Step 1 - Safely Capture File Info ---
# This block runs on every rerun. Safely capture file info from the uploader.
if uploaded_file is not None:
    try:
        file_id_now = uploaded_file.id
        file_name_now = uploaded_file.name
        file_type_now = uploaded_file.type
        file_bytes_now = uploaded_file.getvalue()

        # Check if this file is already processed, in main queue, or already captured
        is_already_processed = file_id_now in st.session_state.processed_file_ids
        is_already_in_main_queue = st.session_state.get('file_to_summarize', None) is not None and st.session_state.file_to_summarize['id'] == file_id_now
        is_already_safely_captured = st.session_state.get('file_info_to_process_safely_captured', None) is not None and st.session_state.file_info_to_process_safely_captured['id'] == file_id_now
        # Also check if it's the image currently waiting to be sent with the next prompt
        is_current_image_for_prompt = st.session_state.get('uploaded_image_for_next_prompt', None) is not None and \
                                      st.session_state.uploaded_image_for_next_prompt.get('id') == file_id_now


        # If it's a new file (not processed, not in queues/capture, not waiting image):
        if not is_already_processed and not is_already_in_main_queue and not is_already_safely_captured and not is_current_image_for_prompt:
            logging.info(f"Detected new file and attempting to safely capture details: {file_name_now} (ID: {file_id_now})")
            st.session_state.file_info_to_process_safely_captured = {
                'id': file_id_now, 'name': file_name_now, 'type': file_type_now, 'bytes': file_bytes_now
            }
            st.rerun() # Trigger next step

        # If it IS already safely captured, clear capture state
        elif is_already_safely_captured and not is_already_processed and not is_already_in_main_queue and not is_current_image_for_prompt:
             logging.info(f"File '{file_name_now}' (ID: {file_id_now}) was already safely captured. Clearing capture state.")
             st.session_state.file_info_to_process_safely_captured = None
             pass # Next processing step will handle it


    except AttributeError as e:
        logging.warning(f"AttributeError caught during uploaded_file attribute access (likely stale object): {e}")
        pass
    except Exception as e:
         logging.error(f"Unexpected error during uploaded_file attribute access: {e}", exc_info=True)
         pass


# --- File Upload Handling and Queuing: Step 2 - Process Captured Info (Bytes to Text or Queue Image) ---
# Process file info captured in Step 1. Convert text files to text, queue images.
if st.session_state.get('file_info_to_process_safely_captured', None) is not None:
    file_info_captured = st.session_state.file_info_to_process_safely_captured

    if file_info_captured['id'] not in st.session_state.processed_file_ids: # Not yet fully processed

        logging.info(f"Processing safely captured file info (Bytes to Text/Queue Image) for '{file_info_captured['name']}' (ID: {file_info_captured['id']}).")
        st.session_state.file_info_to_process_safely_captured = None # Clear captured state

        # Handle image files: store bytes to be sent with the next prompt
        if file_info_captured['type'] in ['image/jpeg', 'image/png']:
             logging.info(f"Queuing image file for next prompt: {file_info_captured['name']}")

             # Store image info in a dedicated state variable for the next prompt
             st.session_state.uploaded_image_for_next_prompt = {
                 'id': file_info_captured['id'],
                 'name': file_info_captured['name'],
                 'type': file_info_captured['type'],
                 'bytes': file_info_captured['bytes'] # Store bytes
             }

             # Display a message to the user that the image is ready to be sent with the next prompt
             st.info(f"✨ 이미지가 업로드되었습니다! 다음 질문과 함께 모델에게 전송됩니다.")

             # Mark image file as processed (as it's now in the 'waiting for prompt' state)
             st.session_state.processed_file_ids.add(file_info_captured['id'])

             # Display the image in the chat history area for visual confirmation
             with st.chat_message("user"): # Display as part of user's intent
                 st.image(file_info_captured['bytes'], caption=f"업로드된 이미지: {file_info_captured['name']}", use_column_width=True)

             st.rerun() # Update UI and wait for user prompt

        else: # Handle text-based files: Read content and queue for summarization
            content_text, read_error = read_file(file_info_captured['bytes'], file_info_captured['name'], file_info_captured['type'])

            if read_error:
                st.error(f"'{file_info_captured['name']}' 파일 읽기 실패: {read_error}")
                st.session_state.processed_file_ids.add(file_info_captured['id']) # Mark as processed (failed read)
            elif not content_text:
                st.warning(f"'{file_info_captured['name']}' 파일 내용이 비어 있습니다. 요약을 건너뜁니다.")
                st.session_state.processed_file_ids.add(file_info_captured['id']) # Mark as processed (empty)
            else: # Text content read successfully
                st.session_state.file_to_summarize = { # Queue for summarization
                    'id': file_info_captured['id'], 'name': file_info_captured['name'], 'content': content_text
                }
                logging.info(f"File '{file_info_captured['name']}' text content queued for summarization.")
                st.rerun() # Trigger summarization step


# --- Main Summarization Processing: Step 3 - Summarize Text ---
# Process text files queued in Step 2.
if st.session_state.get('file_to_summarize', None) is not None and \
   st.session_state.file_to_summarize['id'] not in st.session_state.processed_file_ids: # Not yet fully processed

    file_info_to_process = st.session_state.file_to_summarize
    file_id_to_process = file_info_to_process['id']
    filename_to_process = file_info_to_process['name']
    file_content_to_process = file_info_to_process['content'] # Text content

    st.session_state.file_to_summarize = None # Clear the main queue slot

    logging.info(f"Starting summarization processing from queue: {filename_to_process} (ID: {file_id_to_process}).")

    with st.spinner(f"'{filename_to_process}' 처리 및 요약 중..."):
        tokenizer = get_tokenizer()
        summary, summary_error = summarize_document(file_content_to_process, filename_to_process, MODEL, tokenizer)

        if summary_error:
             st.warning(f"'{filename_to_process}' 요약 중 일부 오류 발생:\n{summary_error}")

        st.session_state.doc_summaries[filename_to_process] = summary
        st.session_state.processed_file_ids.add(file_id_to_process) # Mark as fully processed

    st.success(f"📄 '{filename_to_process}' 업로드 및 요약 완료! 요약 내용이 대화 컨텍스트에 포함됩니다.") # Help text updated
    logging.info(f"Successfully processed and summarized file: {filename_to_process}.")
    st.rerun() # Update UI


# Display summary expander (remains the same)
if st.session_state.doc_summaries:
    with st.expander("📚 업로드된 문서 요약 보기", expanded=False):
        for fname in sorted(st.session_state.doc_summaries.keys()):
             summ = st.session_state.doc_summaries[fname]
             st.text_area(f"요약: {fname}", summ, height=150, key=f"summary_display_{fname}", disabled=True)
        if st.button("문서 요약만 지우기", key="clear_doc_summaries_btn_exp"):
             st.session_state.doc_summaries = {}
             # processed_file_ids related to document summaries could be removed here if needed, but let's keep it simple
             # Clear related processing states
             st.session_state.file_to_summarize = None
             st.session_state.file_info_to_process_safely_captured = None
             st.session_state.uploaded_image_for_next_prompt = None # Clear any pending image too
             logging.info("Document summaries cleared by user.")
             st.rerun()


# Display chat history (remains the same)
st.markdown("---")
st.subheader("대화")

# Display the pending image *before* the chat input if it exists
# This provides visual feedback before the user types the next prompt
if st.session_state.get('uploaded_image_for_next_prompt', None) is not None:
     image_info = st.session_state.uploaded_image_for_next_prompt
     # We already displayed it as a user message in Step 2, no need to display again here


msgs_to_display = [msg for msg in st.session_state.messages if msg['role'] != 'system']

for message in msgs_to_display:
    with st.chat_message(message["role"]):
        # Message content can be a string or a list of parts
        content = message["content"]
        if isinstance(content, str):
            st.markdown(content)
        elif isinstance(content, list):
            # Handle multimodal content
            for part in content:
                if part.get("type") == "text" and "text" in part:
                    st.markdown(part["text"])
                elif part.get("type") == "image_url" and "image_url" in part and "url" in part["image_url"]:
                    # Display images embedded in chat history (if they were saved this way)
                    # Note: Our save_history doesn't save list content yet, but this handles potential future format
                    # For now, images are displayed when uploaded (Step 2)
                    pass # Images are handled in Step 2 display


# ------------------------------------------------------------------
# CHAT INPUT & RESPONSE GENERATION (with Streaming)
# ------------------------------------------------------------------
if prompt := st.chat_input("여기에 메시지를 입력하세요..."):
    # Create the new user message content
    user_message_content: Any # Can be string or list

    # Check if there is an uploaded image waiting to be sent with this prompt
    if st.session_state.get('uploaded_image_for_next_prompt', None) is not None:
        image_info = st.session_state.uploaded_image_for_next_prompt
        logging.info(f"Combining image '{image_info['name']}' (ID: {image_info['id']}) with next prompt.")

        # Construct multimodal content: text part + image_url part
        # Image data needs to be Base64 encoded for the image_url format
        base64_image = base64.b64encode(image_info['bytes']).decode('utf-8')
        image_url = f"data:{image_info['type']};base64,{base64_image}"

        user_message_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]

        # Clear the waiting image state after it's used in the prompt
        st.session_state.uploaded_image_for_next_prompt = None

    else:
        # No pending image, just send text content
        user_message_content = prompt

    # Add the new user message (text or multimodal) to session state
    st.session_state.messages.append({'role': 'user', 'content': user_message_content})

    # Display the new user message in the chat history area
    with st.chat_message("user"):
        # Display content (handles string or list)
        if isinstance(user_message_content, str):
            st.markdown(user_message_content)
        elif isinstance(user_message_content, list):
             for part in user_message_content:
                 if part.get("type") == "text" and "text" in part:
                     st.markdown(part["text"])
                 elif part.get("type") == "image_url" and "image_url" in part and "url" in part["image_url"]:
                      # Re-display the image part if message content is list
                      try:
                          # Decode Base64 back to bytes for displaying
                          image_url = part["image_url"]["url"]
                          # Expecting "data:image/jpeg;base64,..." or similar
                          header, base64_data = image_url.split(',')
                          image_bytes = base64.b64decode(base64_data)
                          # Infer type from header if possible, or use a default
                          image_type = header.split(':')[1].split(';')[0] if ':' in header and ';' in header else 'image/png' # Default if type missing
                          st.image(image_bytes, use_column_width=True) # Display image
                      except Exception as e:
                           logging.error(f"Error displaying image from multimodal message: {e}", exc_info=True)
                           st.warning("⚠️ 이미지 표시 중 오류가 발생했습니다.")


    # --- Context Building ---
    try:
        tokenizer = get_tokenizer()
        current_model_max_tokens = MODEL_CONTEXT_LIMITS.get(MODEL, 4096)

        # Current system prompt (should be the first message in session state)
        current_system_prompt = st.session_state.messages[0] if 'messages' in st.session_state and st.session_state.messages and st.session_state.messages[0]['role'] == 'system' else SYSTEM_PROMPT

        # Note: Token calculation for multimodal messages (with images) is complex
        # and not precisely handled by num_tokens_from_messages.
        # We will rely on the API to handle token counts and catch potential errors.
        # The context trimming logic below works based on *text* token estimation,
        # which might not prevent exceeding limits when images are included.

        # Filter messages for context: System prompt + (Summaries) + (History) + Current User Message
        # The full message list is in st.session_state.messages
        # The last message is the current user prompt (could be multimodal)
        # We need to build the context ensuring total text tokens + some allowance for images
        # stays within a reasonable bound. Since image token calculation is tricky,
        # the existing text-based trimming logic will still be applied to text parts.

        # For multimodal context, the message list sent to the API can contain
        # messages with `content` as a string (text-only) or a list (text+image).
        # The structure of st.session_state.messages now supports this.

        # Build the conversation context list to send to the API
        # Start with system prompt
        conversation_context = [current_system_prompt]
        # Note: tokens_used calculation will only be accurate for text parts

        # Add document summaries (these are text-only system messages)
        doc_summary_context = []
        for fname in sorted(st.session_state.doc_summaries.keys()):
            summ = st.session_state.doc_summaries[fname]
            summary_msg = {'role': 'system', 'content': f"[문서 '{fname}' 요약 참고]\n{summ}"}
            doc_summary_context.append(summary_msg)
        # Add summaries to context (order after system prompt)
        conversation_context.extend(doc_summary_context)


        # Add conversation history (user and assistant messages)
        # Exclude system messages and the current user message (which is the last one)
        history_messages = [msg for msg in st.session_state.messages[:-1] if msg['role'] != 'system'] # Exclude last user message
        # The last message in st.session_state.messages is the current user message (text or multimodal)
        current_user_message = st.session_state.messages[-1]


        # --- Token-based history trimming (primarily based on text tokens) ---
        # Estimate tokens for current user message (only text part counted here)
        current_user_tokens_estimate = num_tokens_from_messages([current_user_message], tokenizer)

        # Estimate tokens for system prompt and summaries
        base_context_tokens_estimate = num_tokens_from_messages(conversation_context, tokenizer)

        available_for_history_estimate = current_model_max_tokens - base_context_tokens_estimate - current_user_tokens_estimate - RESERVED_TOKENS

        history_context = []
        history_tokens_added_estimate = 0

        # Add history messages from most recent, respecting estimated token limit
        for msg in reversed(history_messages):
            msg_tokens_estimate = num_tokens_from_messages([msg], tokenizer)
            if available_for_history_estimate - (history_tokens_added_estimate + msg_tokens_estimate) >= 0:
                 history_context.insert(0, msg) # Add to the beginning to maintain chronological order
                 history_tokens_added_estimate += msg_tokens_estimate
            else:
                 logging.warning("Older chat history skipped due to estimated token limit.")
                 break # Stop adding history


        # Add the trimmed history to the context
        conversation_context.extend(history_context)
        # Finally, add the current user message (which might be multimodal)
        conversation_context.append(current_user_message)

        # Total estimated tokens (includes text from all parts, but NOT image tokens)
        total_estimated_tokens = num_tokens_from_messages(conversation_context, tokenizer)
        logging.info(f"Estimated context tokens (text parts only): {total_estimated_tokens} for model {MODEL}.")
        # Note: The actual token count sent to the API with images will be higher and is calculated by OpenAI.
        # RateLimitError (429) or ContextWindowError (400) might still occur depending on image size and user's plan.


    except Exception as e:
        st.error(f"대화 컨텍스트 구성 중 오류 발생: {e}")
        logging.error(f"Error constructing conversation context: {e}", exc_info=True)
        # Fallback: Use minimal context if building fails
        current_system_prompt = st.session_state.messages[0] if 'messages' in st.session_state and st.session_state.messages and st.session_state.messages[0]['role'] == 'system' else SYSTEM_PROMPT
        # Try to include the last user message if possible, even if it's multimodal
        last_user_msg = st.session_state.messages[-1] if 'messages' in st.session_state and st.session_state.messages else None
        conversation_context = [current_system_prompt]
        if last_user_msg:
             conversation_context.append(last_user_msg)

        # If even minimal context creation fails, or results in only system message
        if not conversation_context or (len(conversation_context) == 1 and conversation_context[0]['role'] == 'system'):
            st.chat_message("assistant").error("⚠️ 최소 컨텍스트 구성에도 실패했습니다. 응답을 생성할 수 없습니다.")
            return # Stop processing this prompt


    # --- API Call and Response Generation (with Streaming) ---
    # Ensure context is valid and contains more than just the system message
    if conversation_context and any(msg['role'] != 'system' for msg in conversation_context):
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Placeholder for streaming
            full_response = ""
            try:
                # Pass the constructed conversation_context (which can contain multimodal messages)
                stream = client.chat.completions.create(
                    model=MODEL, # Use the selected multimodal model
                    messages=conversation_context,
                    stream=True,
                    temperature=0.75 if MODE == 'Poetic' else 0.4,
                    timeout=120
                )
                # Process the streamed response
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        chunk_content = chunk.choices[0].delta.content
                        full_response += chunk_content
                        message_placeholder.markdown(full_response + "▌") # Update streaming response

                message_placeholder.markdown(full_response) # Display final response
                logging.info(f"Assistant response received (length: {len(full_response)} chars).")

            except Exception as e:
                full_response = f"⚠️ 죄송합니다, 응답 생성 중 오류가 발생했습니다: {e}"
                message_placeholder.error(full_response)
                logging.error(f"Error during OpenAI API call/streaming: {e}", exc_info=True)
                # Check for specific errors like RateLimitError or ContextWindowError
                if "rate_limit_exceeded" in str(e).lower():
                     st.error("⚠️ Rate Limit Exceeded: OpenAI API 사용량 제한을 초과했습니다. 대화 길이를 줄이거나 잠시 후 다시 시도하세요.")
                elif "context_window_exceeded" in str(e).lower():
                     st.error("⚠️ Context Window Exceeded: 대화 컨텍스트 길이가 모델의 최대 한도를 초과했습니다. 대화 길이나 업로드된 문서 양을 줄이세요.")


    else:
         # Should not happen if context building fallback works, but as a safeguard
         full_response = "⚠️ 응답 생성에 필요한 유효한 대화 컨텍스트가 없습니다."
         st.chat_message("assistant").error(full_response)


    # Add the assistant's response to session state
    if full_response:
         st.session_state.messages.append({'role': 'assistant', 'content': full_response})
         # Save history (save_history excludes system/error messages)
         save_history(HISTORY_FILE, st.session_state.messages)


# --- Footer or additional info ---
st.sidebar.markdown("---")
st.sidebar.caption("Liel Chatbot v1.7 (멀티모달 이미지 지원)") # 버전 및 상태 업데이트
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
from typing import List, Dict, Tuple, Optional, Any
import io
import base64

# ------------------------------------------------------------------
# 로깅 설정
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# ------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------
MODEL_CONTEXT_LIMITS = {
    "gpt-3.5-turbo": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
}
MULTIMODAL_VISION_MODELS = ["gpt-4o", "gpt-4-turbo"] # 멀티모달 지원 모델 목록

DEFAULT_ENCODING = "cl100k_base"
CHUNK_SIZE = 2000
RESERVED_TOKENS = 1500
HISTORY_FILE = "history/chat_history.json" # 기록 파일 경로에 디렉토리 포함
CHUNK_PROMPT_FOR_SUMMARY = 'Summarize the key points of this text chunk in 2-3 concise bullet points, focusing on the main information.'

# ------------------------------------------------------------------
# STREAMLIT PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="Liel – AI Chatbot", layout="wide", initial_sidebar_state="auto")

# ------------------------------------------------------------------
# OPENAI CLIENT INITIALIZATION & API KEY HANDLING
# ------------------------------------------------------------------
def initialize_openai_client() -> Optional[OpenAI]:
    """Streamlit Secrets 또는 환경 변수에서 API 키를 로드하여 OpenAI 클라이언트를 초기화합니다."""
    api_key = None
    try:
        # 1. Streamlit Secrets 확인 (클라우드 배포 시 권장)
        if "general" in st.secrets and "OPENAI_API_KEY" in st.secrets["general"]:
            api_key = st.secrets["general"]["OPENAI_API_KEY"]
            logging.info("API Key loaded from Streamlit Secrets.")
        # 2. 환경 변수 확인 (Secrets에 없거나 로컬 실행 시)
        elif os.environ.get("OPENAI_API_KEY"):
            api_key = os.environ.get("OPENAI_API_KEY")
            logging.info("API Key loaded from environment variable.")

    except Exception as e:
        logging.warning(f"Error accessing API key from secrets or env: {e}")

    if not api_key or not api_key.startswith("sk-"):
        st.error("❌ OpenAI API 키를 찾을 수 없거나 형식이 올바르지 않습니다.")
        st.warning("API 키를 `.streamlit/secrets.toml` 파일에 `[general]\nOPENAI_API_KEY='sk-...'` 형식으로 저장하거나, 환경 변수로 설정해주세요.")
        return None

    try:
        client = OpenAI(api_key=api_key)
        logging.info("OpenAI client initialized.")
        return client
    except Exception as e:
        st.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
        logging.error(f"Client initialization failed: {e}", exc_info=True)
        return None

client = initialize_openai_client()

if client is None:
    st.stop() # 클라이언트 초기화 실패 시 앱 중지

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """지정된 이름의 tiktoken 인코더를 반환합니다."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except ValueError:
        logging.warning(f"Encoding '{encoding_name}' not found. Using default '{DEFAULT_ENCODING}'.")
        return tiktoken.get_encoding(DEFAULT_ENCODING)

def num_tokens_from_string(string: str, encoding: tiktoken.Encoding) -> int:
    """주어진 문자열의 토큰 수를 반환합니다."""
    return len(encoding.encode(string)) if string else 0

def num_tokens_from_messages(messages: List[Dict[str, Any]], encoding: tiktoken.Encoding) -> int:
    """Calculates tokens for text parts in messages (simplified for multimodal)."""
    num_tokens = 0
    for message in messages:
        num_tokens += 4 # base tokens per message

        content = message.get("content")
        if isinstance(content, str):
             num_tokens += num_tokens_from_string(content, encoding)
        elif isinstance(content, list):
             for part in content:
                 if part.get("type") == "text" and "text" in part:
                      num_tokens += num_tokens_from_string(part["text"], encoding)
                 # Image tokens are NOT calculated here.

        if "name" in message:
            num_tokens -= 1

    num_tokens += 2 # assistant reply start tokens
    return num_tokens


# read_file remains the same
@st.cache_data(show_spinner=False, hash_funcs={docx.document.Document: id, pd.DataFrame: pd.util.hash_pandas_object})
def read_file(uploaded_file_content_bytes, filename, file_type) -> Tuple[str, Optional[str]]:
    """업로드된 파일의 바이트 내용을 읽어 텍스트 반환 (에러 처리 포함)."""
    try:
        logging.info(f"Reading file: {filename} (Type: {file_type})")
        file_like_object = io.BytesIO(uploaded_file_content_bytes)

        if file_type == 'text/plain':
            try:
                return file_like_object.read().decode('utf-8'), None
            except UnicodeDecodeError:
                file_like_object.seek(0)
                return file_like_object.read().decode('cp949'), None
        elif file_type == 'application/pdf':
            reader = PdfReader(file_like_object)
            text_parts = [page.extract_text() or '' for page in reader.pages]
            return '\n'.join(part for part in text_parts if part), None
        elif 'wordprocessingml.document' in file_type:
            doc = docx.Document(file_like_object)
            return '\n'.join(p.text for p in doc.paragraphs), None
        elif 'spreadsheetml.sheet' in file_type:
            df = pd.read_excel(file_like_object, engine='openpyxl')
            return df.to_csv(index=False, sep='\t'), None
        elif file_type in ['image/jpeg', 'image/png']:
             return '', f"이미지 파일은 텍스트 변환 지원 안 함: {file_type}"
        else:
            return '', f"지원하지 않는 파일 형식: {file_type}"
    except Exception as e:
        logging.error(f"Error reading file {filename}: {e}", exc_info=True)
        return '', f"파일 처리 중 오류 발생: {e}"


# summarize_document remains the same
@st.cache_data(show_spinner=False)
def summarize_document(text: str, filename: str, model: str, tokenizer: tiktoken.Encoding) -> Tuple[str, Optional[str]]:
    """주어진 텍스트를 청크로 나누어 모델로 요약."""
    if not text:
        return "(문서 내용 없음)", None

    summaries = []
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    total_chunks = len(chunks)
    logging.info(f"Starting summarization for '{filename}' ({total_chunks} chunks).")

    progress_bar = st.progress(0, text=f"'{filename}' 요약 중... [0/{total_chunks}]")
    summary_errors = []

    for i, chunk in enumerate(chunks):
        progress = (i + 1) / total_chunks
        progress_bar.progress(progress, text=f"'{filename}' 요약 중... [{i+1}/{total_chunks}]")

        chunk_tokens = num_tokens_from_string(chunk, tokenizer)
        model_limit = MODEL_CONTEXT_LIMITS.get(model, 8192)
        if chunk_tokens > model_limit - 500:
             logging.warning(f"Chunk {i+1} tokens ({chunk_tokens}) near/exceeds model limit ({model}).")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': CHUNK_PROMPT_FOR_SUMMARY},
                    {'role': 'user', 'content': chunk}
                ],
                max_tokens=250,
                temperature=0.3,
                timeout=60
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                summaries.append(response.choices[0].message.content.strip())
            else:
                 logging.warning(f"Chunk {i+1} summarization returned no content.")
                 summaries.append(f"(청크 {i+1} 요약 내용 없음)")

            sleep(0.15)
        except Exception as e:
            error_msg = f"청크 {i+1} 요약 중 오류 발생: {e}"
            st.warning(error_msg)
            logging.error(f"Error summarizing chunk {i+1}: {e}", exc_info=True)
            summaries.append(f"(청크 {i+1} 요약 실패)")
            summary_errors.append(error_msg)

    progress_bar.empty()
    return '\n'.join(summaries), "\n".join(summary_errors) if summary_errors else None

# load_history remains the same
@st.cache_data(show_spinner=False)
def load_history(path: str) -> List[Dict[str, str]]:
    """JSON 파일에서 대화 기록을 로드합니다."""
    if not os.path.exists(path):
        logging.info(f"History file not found at {path}.")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            history = json.load(f)
            logging.info(f"Loaded {len(history)} messages from {path}.")
            return history
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        logging.warning(f"Error loading history file {path}: {e}")
        if os.path.exists(path):
             st.sidebar.warning(f"대화 기록 파일을 읽는 중 오류 발생: {e}. 새 기록으로 시작합니다. ({os.path.basename(path)})")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading history from {path}: {e}", exc_info=True)
        st.sidebar.error(f"대화 기록 로드 중 예상치 못한 오류 발생: {e}")
        return []


# save_history remains the same (only saves text messages)
def save_history(path: str, msgs: List[Dict[str, Any]]):
    """대화 기록을 JSON 파일에 저장합니다 (시스템 메시지 및 멀티모달 콘텐츠 제외)."""
    # 현재 save_history는 content가 문자열인 메시지만 저장하도록 필터링합니다.
    # 멀티모달 메시지를 저장하려면 저장 형식을 고민해야 합니다.
    # 여기서는 기존 save_history 로직 유지 (텍스트 메시지만 저장)
    msgs_to_save = [msg for msg in msgs if msg['role'] != 'system' and isinstance(msg.get('content'), str)]

    if not msgs_to_save:
        if os.path.exists(path):
            try:
                os.remove(path)
                logging.info(f"History file {path} removed as messages are empty.")
            except OSError as e:
                logging.error(f"Failed to remove empty history file {path}: {e}")
        return

    history_dir = os.path.dirname(path)
    if history_dir:
        try:
            os.makedirs(history_dir, exist_ok=True)
        except Exception as e:
             logging.error(f"Error creating history directory {history_dir}: {e}", exc_info=True)
             st.sidebar.error(f"기록 폴더 생성 실패: {e}. 기록 저장이 안 될 수 있습니다.")
             return

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(msgs_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error saving history to {path}: {e}", exc_info=True)


# ------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------
if 'messages' not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = load_history(HISTORY_FILE)
    logging.info(f"Session initialized. Loaded {len(st.session_state.messages)} messages.")

if 'doc_summaries' not in st.session_state:
    st.session_state.doc_summaries: Dict[str, str] = {}
if 'processed_file_ids' not in st.session_state:
    st.session_state.processed_file_ids: set = set()

if 'file_to_summarize' not in st.session_state:
    st.session_state.file_to_summarize: Optional[Dict] = None

if 'file_info_to_process_safely_captured' not in st.session_state:
     st.session_state.file_info_to_process_safely_captured: Optional[Dict] = None

# State for image data waiting for the next prompt
if 'uploaded_image_for_next_prompt' not in st.session_state:
    st.session_state.uploaded_image_for_next_prompt: Optional[Dict] = None


# ------------------------------------------------------------------
# SIDEBAR: MODEL, MODE SELECTION & OPTIONS
# ------------------------------------------------------------------
st.sidebar.title("⚙️ 설정")

MODEL = st.sidebar.selectbox(
    '모델 선택 (멀티모달 지원)',
    MULTIMODAL_VISION_MODELS,
    index=MULTIMODAL_VISION_MODELS.index("gpt-4o") if "gpt-4o" in MULTIMODAL_VISION_MODELS else 0
)
MAX_CONTEXT_TOKENS = MODEL_CONTEXT_LIMITS.get(MODEL, 4096)

MODE = st.sidebar.radio('응답 모드', ('Poetic', 'Logical'), index=0, key='mode_selection')

st.sidebar.markdown("---")
st.sidebar.subheader("관리")

# build_full_session_content remains the same
def build_full_session_content() -> str:
    """문서 요약과 전체 대화 기록을 합쳐 텍스트로 만듭니다."""
    parts = []
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    parts.append(f"Liel Chat Session Content - {timestamp}")
    parts.append(f"Model: {MODEL}, Mode: {MODE}\n")

    if st.session_state.doc_summaries:
        parts.append("===== Uploaded Document Summaries =====")
        for fname in sorted(st.session_state.doc_summaries.keys()):
            summ = st.session_state.doc_summaries[fname]
            parts.append(f"\n--- Summary: {fname} ---")
            parts.append(summ)
            parts.append("-" * (len(fname) + 16))
        parts.append("\n" + "=" * 30 + "\n")

    parts.append("===== Conversation History =====")
    msgs_to_include = [msg for msg in st.session_state.messages if msg['role'] != 'system']
    if not msgs_to_include:
         parts.append("(No conversation yet)")
    else:
        for m in msgs_to_include:
            role_icon = "👤 User" if m['role'] == 'user' else "🤖 Liel"
            parts.append(f"\n{role_icon}:\n")
            content = m.get('content')
            if isinstance(content, str):
                 parts.append(content)
            elif isinstance(content, list):
                 multimodal_parts = []
                 for part in content:
                     if part.get("type") == "text" and "text" in part:
                         multimodal_parts.append(part["text"])
                     elif part.get("type") == "image_url" and "image_url" in part:
                         multimodal_parts.append("[Uploaded Image]")
                 parts.append("\n".join(multimodal_parts))
            else:
                 parts.append("(Unsupported message content format)")

            parts.append("-" * 20)

    return '\n'.join(parts)


# Download Button remains the same
if [msg for msg in st.session_state.messages if msg['role'] != 'system'] or st.session_state.doc_summaries:
    session_content_txt = build_full_session_content()
    download_filename = f"liel_session_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"
    st.sidebar.download_button(
        label="📥 현재 세션 내용 다운로드",
        data=session_content_txt.encode('utf-8'),
        file_name=download_filename,
        mime='text/plain',
        help="업로드된 문서 요약과 현재까지의 대화 기록 전체를 텍스트 파일로 다운로드합니다."
    )

# Clear Button remains the same
if st.sidebar.button("🔄 대화 및 문서 요약 초기화"):
    st.session_state.messages = []
    st.session_state.doc_summaries = {}
    st.session_state.processed_file_ids = set()
    st.session_state.file_to_summarize = None
    st.session_state.file_info_to_process_safely_captured = None
    st.session_state.uploaded_image_for_next_prompt = None

    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            logging.info(f"History file {HISTORY_FILE} removed.")
            st.sidebar.success("대화 기록 파일이 삭제되었습니다.")
        except OSError as e:
            logging.error(f"Failed to remove history file {HISTORY_FILE}: {e}")
            st.sidebar.error(f"기록 파일 삭제 실패: {e}")
    st.rerun()


# ------------------------------------------------------------------
# SYSTEM PROMPT DEFINITION
# ------------------------------------------------------------------
SYSTEM_PROMPT_CONTENT = (
    'You are Liel, a poetic, emotionally intelligent chatbot with lyrical grace. Respond with warmth, creativity, and empathy. Use rich language and metaphors when appropriate.'
    if MODE == 'Poetic' else
    'You are Liel, a highly analytical assistant focused on logic and precision. Provide clear, structured, and concise answers. Use bullet points or numbered lists for clarity when needed.'
)
SYSTEM_PROMPT = {'role': 'system', 'content': SYSTEM_PROMPT_CONTENT}

if 'messages' in st.session_state:
     st.session_state.messages = [msg for msg in st.session_state.messages if msg['role'] != 'system']
     st.session_state.messages.insert(0, SYSTEM_PROMPT)


# ------------------------------------------------------------------
# UI HEADER
# ------------------------------------------------------------------
st.title(f'💬 Liel – {MODE} Chatbot')
st.caption(
    "기억과 감정으로 빛나는 당신의 대화 상대, Liel입니다. 파일을 업로드하거나 메시지를 입력하세요."
    if MODE == 'Poetic' else
    "분석적이고 논리적인 대화 상대, Liel입니다. 파일을 업로드하거나 질문해주세요."
)

# ------------------------------------------------------------------
# FILE UPLOAD UI
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    '파일 업로드 (txt, pdf, docx, xlsx, jpg, png)',
    type=['txt', 'pdf', 'docx', 'xlsx', 'jpg', 'png'],
    key="file_uploader",
    help="텍스트, PDF, 워드, 엑셀 파일은 요약하여 컨텍스트에 포함하고, 이미지 파일은 다음 질문과 함께 모델에게 전송합니다."
)

# --- File Upload Handling and Queuing: Step 1 - Safely Capture File Info ---
logging.info(f"--- Start Streamlit Rerun ---") # Log start of rerun
logging.info(f"Uploaded file state: {uploaded_file is not None}") # Log uploader state

if uploaded_file is not None:
    logging.info(f"Step 1: uploaded_file is NOT None. Attempting to capture details for {uploaded_file.name}.")
    try:
        file_id_now = uploaded_file.id
        file_name_now = uploaded_file.name
        file_type_now = uploaded_file.type
        file_bytes_now = uploaded_file.getvalue()
        logging.info(f"Step 1: Successfully got file details - ID: {file_id_now}, Name: {file_name_now}, Type: {file_type_now}.")


        is_already_processed = file_id_now in st.session_state.processed_file_ids
        is_already_in_main_queue = st.session_state.get('file_to_summarize', None) is not None and st.session_state.file_to_summarize['id'] == file_id_now
        is_already_safely_captured = st.session_state.get('file_info_to_process_safely_captured', None) is not None and st.session_state.file_info_to_process_safely_captured['id'] == file_id_now
        is_current_image_for_prompt = st.session_state.get('uploaded_image_for_next_prompt', None) is not None and st.session_state.uploaded_image_for_next_prompt.get('id') == file_id_now

        logging.info(f"Step 1: File status checks for {file_name_now}: processed={is_already_processed}, in_queue={is_already_in_main_queue}, safely_captured={is_already_safely_captured}, waiting_image={is_current_image_for_prompt}")


        if not is_already_processed and not is_already_in_main_queue and not is_already_safely_captured and not is_current_image_for_prompt:
            logging.info(f"Step 1: File {file_name_now} is new and needs capture. Storing in session state.")
            st.session_state.file_info_to_process_safely_captured = {
                'id': file_id_now, 'name': file_name_now, 'type': file_type_now, 'bytes': file_bytes_now
            }
            logging.info("Step 1: Triggering rerun after capture.")
            st.rerun()
        elif is_already_safely_captured and not is_already_processed and not is_already_in_main_queue and not is_current_image_for_prompt:
             logging.info(f"Step 1: File '{file_name_now}' already safely captured. Clearing capture state.")
             st.session_state.file_info_to_process_safely_captured = None
             # No rerun here, expecting Step 2 to process it in this or next rerun
             pass
        else:
             logging.info(f"Step 1: File {file_name_now} already in a processing state. Skipping capture.")


    except AttributeError as e:
        logging.warning(f"Step 1: AttributeError caught for {uploaded_file.name if uploaded_file else 'None object'} during attribute access: {e}")
        pass
    except Exception as e:
         logging.error(f"Step 1: Unexpected error during uploaded_file access for {uploaded_file.name if uploaded_file else 'None object'}: {e}", exc_info=True)
         pass
else:
     logging.info("Step 1: uploaded_file is None.")


# --- File Upload Handling and Queuing: Step 2 - Process Captured Info (Bytes to Text or Queue Image) ---
# Process file info captured in Step 1. Convert text files to text, queue images.
if st.session_state.get('file_info_to_process_safely_captured', None) is not None:
    file_info_captured = st.session_state.file_info_to_process_safely_captured
    logging.info(f"Step 2: file_info_to_process_safely_captured is NOT None. Processing {file_info_captured['name']} (ID: {file_info_captured['id']}).")


    if file_info_captured['id'] not in st.session_state.processed_file_ids: # Not yet fully processed
        logging.info(f"Step 2: File {file_info_captured['name']} is not processed. Clearing captured state.")
        st.session_state.file_info_to_process_safely_captured = None # Clear captured state


        # Handle image files: store bytes to be sent with the next prompt
        if file_info_captured['type'] in ['image/jpeg', 'image/png']:
             logging.info(f"Step 2: File {file_info_captured['name']} is an image. Queuing for next prompt.")

             st.session_state.uploaded_image_for_next_prompt = {
                 'id': file_info_captured['id'],
                 'name': file_info_captured['name'],
                 'type': file_info_captured['type'],
                 'bytes': file_info_captured['bytes']
             }

             st.info(f"✨ 이미지가 업로드되었습니다! 다음 질문과 함께 모델에게 전송됩니다.")
             st.session_state.processed_file_ids.add(file_info_captured['id'])

             # Display the image in the chat history area immediately
             logging.info(f"Step 2: Displaying image {file_info_captured['name']} in chat message.")
             with st.chat_message("user"):
                 st.image(file_info_captured['bytes'], caption=f"업로드된 이미지: {file_info_captured['name']}", use_column_width=True)

             logging.info("Step 2: Triggering rerun after queuing image.")
             st.rerun()

        else: # Handle text-based files: Read content and queue for summarization
            logging.info(f"Step 2: File {file_info_captured['name']} is text-based. Reading content.")
            content_text, read_error = read_file(file_info_captured['bytes'], file_info_captured['name'], file_info_captured['type'])

            if read_error:
                st.error(f"'{file_info_captured['name']}' 파일 읽기 실패: {read_error}")
                st.session_state.processed_file_ids.add(file_info_captured['id'])
            elif not content_text:
                st.warning(f"'{file_info_captured['name']}' 파일 내용이 비어 있습니다. 요약을 건너뜁니다.")
                st.session_state.processed_file_ids.add(file_info_captured['id'])
            else:
                st.session_state.file_to_summarize = {
                    'id': file_info_captured['id'], 'name': file_info_captured['name'], 'content': content_text
                }
                logging.info(f"File '{file_info_captured['name']}' text content queued for summarization.")
                logging.info("Step 2: Triggering rerun after queuing text for summarization.")
                st.rerun()
    else:
        logging.info(f"Step 2: File {file_info_captured['name']} (ID: {file_info_captured['id']}) already processed. Skipping Step 2.")

else:
    logging.info("Step 2: file_info_to_process_safely_captured is None.")


# --- Main Summarization Processing: Step 3 - Summarize Text ---
if st.session_state.get('file_to_summarize', None) is not None and st.session_state.file_to_summarize['id'] not in st.session_state.processed_file_ids:

    file_info_to_process = st.session_state.file_to_summarize
    file_id_to_process = file_info_to_process['id']
    filename_to_process = file_info_to_process['name']
    file_content_to_process = file_info_to_process['content']

    logging.info(f"Step 3: file_to_summarize is NOT None. Starting summarization for {filename_to_process} (ID: {file_id_to_process}).")
    st.session_state.file_to_summarize = None # Clear the main queue slot

    with st.spinner(f"'{filename_to_process}' 처리 및 요약 중..."):
        tokenizer = get_tokenizer()
        summary, summary_error = summarize_document(file_content_to_process, filename_to_process, MODEL, tokenizer)

        if summary_error:
             st.warning(f"'{filename_to_process}' 요약 중 일부 오류 발생:\n{summary_error}")

        st.session_state.doc_summaries[filename_to_process] = summary
        st.session_state.processed_file_ids.add(file_id_to_process)

    st.success(f"📄 '{filename_to_process}' 업로드 및 요약 완료! 요약 내용이 대화 컨텍스트에 포함됩니다.")
    logging.info(f"Successfully processed and summarized: {filename_to_process}.")
    logging.info("Step 3: Triggering rerun after summarization.")
    st.rerun()

else:
    logging.info("Step 3: file_to_summarize is None or already processed.")


# Display summary expander
if st.session_state.doc_summaries:
    with st.expander("📚 업로드된 문서 요약 보기", expanded=False):
        for fname in sorted(st.session_state.doc_summaries.keys()):
             summ = st.session_state.doc_summaries[fname]
             st.text_area(f"요약: {fname}", summ, height=150, key=f"summary_display_{fname}", disabled=True)
        if st.button("문서 요약만 지우기", key="clear_doc_summaries_btn_exp"):
             st.session_state.doc_summaries = {}
             st.session_state.processed_file_ids = set() # Clear all processed IDs for simplicity on clear
             st.session_state.file_to_summarize = None
             st.session_state.file_info_to_process_safely_captured = None
             st.session_state.uploaded_image_for_next_prompt = None
             logging.info("Document summaries cleared by user.")
             st.rerun()


# Display chat history
st.markdown("---")
st.subheader("대화")

msgs_to_display = [msg for msg in st.session_state.messages if msg['role'] != 'system']

for message in msgs_to_display:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, str):
            st.markdown(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text" and "text" in part:
                    st.markdown(part["text"])
                elif part.get("type") == "image_url" and "image_url" in part and "url" in part["image_url"]:
                      try:
                          image_url = part["image_url"]["url"]
                          header, base64_data = image_url.split(',')
                          image_bytes = base64.b64decode(base64_data)
                          image_type = header.split(':')[1].split(';')[0] if ':' in header and ';' in header else 'image/png'
                          st.image(image_bytes, use_column_width=True)
                      except Exception as e:
                           logging.error(f"Error displaying image from multimodal message in history: {e}", exc_info=True)
                           st.warning("⚠️ 이미지 표시 중 오류가 발생했습니다.")


# ------------------------------------------------------------------
# CHAT INPUT & RESPONSE GENERATION (with Streaming)
# ------------------------------------------------------------------
if prompt := st.chat_input("여기에 메시지를 입력하세요..."):
    logging.info(f"Chat input detected: '{prompt}'")
    user_message_content: Any

    # Check if there is an uploaded image waiting to be sent with this prompt
    if st.session_state.get('uploaded_image_for_next_prompt', None) is not None:
        image_info = st.session_state.uploaded_image_for_next_prompt
        logging.info(f"Combining image '{image_info['name']}' (ID: {image_info['id']}) with current prompt.")

        # Encode image bytes to Base64
        try:
            base64_image = base64.b64encode(image_info['bytes']).decode('utf-8')
            image_url = f"data:{image_info['type']};base64,{base64_image}"

            # Construct multimodal content: text part + image_url part
            user_message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
            logging.info(f"Constructed multimodal message content for {image_info['name']} and prompt.")

        except Exception as e:
            logging.error(f"Error encoding or formatting image for multimodal message: {e}", exc_info=True)
            st.error("⚠️ 이미지를 메시지에 포함하는 중 오류가 발생했습니다.")
            user_message_content = prompt # Send text only if image processing fails
            # Do NOT clear uploaded_image_for_next_prompt state if it failed? Or maybe clear to prevent retrying?
            # Let's clear it to avoid getting stuck.
            st.session_state.uploaded_image_for_next_prompt = None


        # Clear the waiting image state after it's used (even if encoding failed, to avoid infinite loop)
        st.session_state.uploaded_image_for_next_prompt = None
        logging.info("Cleared uploaded_image_for_next_prompt state.")

    else:
        # No pending image, just send text content
        logging.info("No pending image. Sending text-only message.")
        user_message_content = prompt

    # Add the new user message (text or multimodal) to session state
    st.session_state.messages.append({'role': 'user', 'content': user_message_content})
    logging.info("Added user message to session state.")

    # Display the new user message in the chat history area
    with st.chat_message("user"):
        content = user_message_content
        if isinstance(content, str):
            st.markdown(content)
        elif isinstance(content, list):
             # Display multimodal parts
             for part in content:
                 if part.get("type") == "text" and "text" in part:
                     st.markdown(part["text"])
                 elif part.get("type") == "image_url" and "image_url" in part and "url" in part["image_url"]:
                      # Re-display the image part
                      try:
                          image_url = part["image_url"]["url"]
                          header, base64_data = image_url.split(',')
                          image_bytes = base64.b64decode(base64_data)
                          image_type = header.split(':')[1].split(';')[0] if ':' in header and ';' in header else 'image/png'
                          st.image(image_bytes, use_column_width=True)
                      except Exception as e:
                           logging.error(f"Error displaying image from multimodal message in chat area: {e}", exc_info=True)
                           st.warning("⚠️ 이미지 표시 중 오류가 발생했습니다.")

    # --- Context Building ---
    should_proceed_with_api_call = True
    conversation_context: List[Dict[str, Any]] = [] # Initialize conversation_context

    try:
        tokenizer = get_tokenizer()
        current_model_max_tokens = MODEL_CONTEXT_LIMITS.get(MODEL, 4096)

        current_system_prompt = st.session_state.messages[0] if 'messages' in st.session_state and st.session_state.messages and st.session_state.messages[0]['role'] == 'system' else SYSTEM_PROMPT

        base_tokens_estimate = num_tokens_from_messages([current_system_prompt], tokenizer)

        # Estimate tokens for current user message (text parts only)
        current_user_tokens_estimate = num_tokens_from_messages([st.session_state.messages[-1]], tokenizer) # Use the message just added

        available_tokens_for_context_estimate = current_model_max_tokens - base_tokens_estimate - current_user_tokens_estimate - RESERVED_TOKENS

        if available_tokens_for_context_estimate <= 0:
             st.error("설정된 모델의 컨텍스트 길이와 예약 토큰으로 인해 대화 컨텍스트를 구성할 수 없습니다. 모델 변경 또는 예약 토큰 감소를 고려하세요.")
             logging.error("Context window too small based on text token estimate.")
             should_proceed_with_api_call = False

        if should_proceed_with_api_call:
            conversation_context = [current_system_prompt]

            doc_summary_context = []
            doc_tokens_added_estimate = 0
            for fname in sorted(st.session_state.doc_summaries.keys()):
                summ = st.session_state.doc_summaries[fname]
                summary_msg = {'role': 'system', 'content': f"[문서 '{fname}' 요약 참고]\n{summ}"}
                summary_tokens_estimate = num_tokens_from_messages([summary_msg], tokenizer)

                if available_tokens_for_context_estimate - (doc_tokens_added_estimate + summary_tokens_estimate) >= 0:
                    doc_summary_context.append(summary_msg)
                    doc_tokens_added_estimate += summary_tokens_estimate
                else:
                    logging.warning(f"Document summary '{fname}' skipped due to estimated token limit.")
                    break

            conversation_context.extend(doc_summary_context)

            history_messages = [msg for msg in st.session_state.messages[:-1] if msg['role'] != 'system']
            history_context = []
            history_tokens_added_estimate = 0

            for msg in reversed(history_messages):
                msg_tokens_estimate = num_tokens_from_messages([msg], tokenizer)
                if available_tokens_for_context_estimate - (doc_tokens_added_estimate + history_tokens_added_estimate + msg_tokens_estimate) >= 0:
                     history_context.insert(0, msg)
                     history_tokens_added_estimate += msg_tokens_estimate
                else:
                     logging.warning("Older chat history skipped due to estimated token limit.")
                     break

            conversation_context.extend(history_context)
            # Add the current user message at the end
            conversation_context.append(st.session_state.messages[-1])

            total_estimated_tokens = num_tokens_from_messages(conversation_context, tokenizer)
            logging.info(f"Estimated final context tokens (text only): {total_estimated_tokens} for model {MODEL}.")


    except Exception as e:
        st.error(f"대화 컨텍스트 구성 중 오류 발생: {e}")
        logging.error(f"Error constructing conversation context: {e}", exc_info=True)
        should_proceed_with_api_call = False

        # Ensure conversation_context has at least system + current user message for fallback display
        current_system_prompt = st.session_state.messages[0] if 'messages' in st.session_state and st.session_state.messages and st.session_state.messages[0]['role'] == 'system' else SYSTEM_PROMPT
        conversation_context = [current_system_prompt]
        if 'messages' in st.session_state and st.session_state.messages:
             conversation_context.append(st.session_state.messages[-1]) # Add the last user message


    # --- API Call and Response Generation (with Streaming) ---
    if should_proceed_with_api_call and conversation_context and any(msg['role'] != 'system' for msg in conversation_context):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                logging.info(f"Calling OpenAI API with model {MODEL}.")
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=conversation_context,
                    stream=True,
                    temperature=0.75 if MODE == 'Poetic' else 0.4,
                    timeout=120
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        chunk_content = chunk.choices[0].delta.content
                        full_response += chunk_content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                logging.info(f"Assistant response received (length: {len(full_response)} chars).")

            except Exception as e:
                full_response = f"⚠️ 죄송합니다, 응답 생성 중 오류가 발생했습니다: {e}"
                message_placeholder.error(full_response)
                logging.error(f"Error during OpenAI API call/streaming: {e}", exc_info=True)
                if "rate_limit_exceeded" in str(e).lower():
                     st.error("⚠️ Rate Limit Exceeded: OpenAI API 사용량 제한을 초과했습니다. 대화 길이를 줄이거나 잠시 후 다시 시도하세요.")
                elif "context_window_exceeded" in str(e).lower():
                     st.error("⚠️ Context Window Exceeded: 대화 컨텍스트 길이가 모델의 최대 한도를 초과했습니다. 대화 길이나 업로드된 문서 양을 줄이세요.")


    else:
         if should_proceed_with_api_call: # If context building succeeded but context is somehow invalid
              full_response = "⚠️ 응답 생성에 필요한 유효한 대화 컨텍스트가 없습니다. 오류 로그를 확인하세요."
              st.chat_message("assistant").error(full_response)
         else:
              # Error message was already displayed in the context building block
              pass # Do nothing more if should_proceed_with_api_call is False


    if full_response:
         st.session_state.messages.append({'role': 'assistant', 'content': full_response})
         save_history(HISTORY_FILE, st.session_state.messages)


# --- Footer or additional info ---
st.sidebar.markdown("---")
st.sidebar.caption("Liel Chatbot v1.7.2 (로깅 강화)") # 버전 및 로깅 상태 업데이트
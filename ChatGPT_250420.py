import json
import os
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import docx
import pandas as pd
from time import sleep # summarize_document 함수에서 사용
import tiktoken
import logging
from typing import List, Dict, Tuple, Optional
import io # BytesIO를 read_file 함수에서 사용

# ------------------------------------------------------------------
# 로깅 설정
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, # 로그 레벨 설정 (INFO, WARNING, ERROR 등)
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
}
DEFAULT_ENCODING = "cl100k_base"
CHUNK_SIZE = 2000 # summarize_document 함수에서 사용
RESERVED_TOKENS = 1500 # 컨텍스트 관리에 사용
HISTORY_FILE = "chat_history.json"
CHUNK_PROMPT_FOR_SUMMARY = 'Summarize the key points of this text chunk in 2-3 concise bullet points, focusing on the main information.' # summarize_document 함수에서 사용

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
    # 1. Streamlit Secrets 확인 (클라우드 배포 시 권장)
    try:
        if "general" in st.secrets and "OPENAI_API_KEY" in st.secrets["general"]:
            api_key = st.secrets["general"]["OPENAI_API_KEY"]
            logging.info("OpenAI API Key loaded from Streamlit Secrets.")
        else:
             logging.info("OpenAI API Key not found in Streamlit Secrets.")
    except Exception as e:
        # secrets 속성 접근 시 에러가 날 수 있습니다.
        logging.warning(f"Could not read Streamlit Secrets['general']['OPENAI_API_KEY']: {e}")

    # 2. 환경 변수 확인 (Secrets에 없거나 로컬 실행 시)
    if not api_key:
        # os.getenv 대신 os.environ.get 사용 권장 (같은 기능이지만 명시적)
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            logging.info("OpenAI API Key loaded from environment variable.")
        else:
            logging.warning("OpenAI API Key not found in environment variables either.")

    if not api_key or not api_key.startswith("sk-"):
        st.error("❌ OpenAI API 키를 찾을 수 없거나 형식이 올바르지 않습니다.")
        st.warning("API 키를 `.streamlit/secrets.toml` 파일에 `[general]\nOPENAI_API_KEY='sk-...'` 형식으로 저장하거나, 환경 변수로 설정해주세요.")
        return None

    try:
        client = OpenAI(api_key=api_key)
        # 간단한 테스트 호출 (선택 사항, 키 유효성 검증)
        # client.models.list()
        logging.info("OpenAI client initialized successfully.")
        return client
    except Exception as e:
        st.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
        logging.error(f"OpenAI client initialization failed: {e}", exc_info=True)
        return None

client = initialize_openai_client()

if client is None:
    st.stop() # 클라이언트 초기화 실패 시 앱 중지

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
# @st.cache_data # tiktoken 로딩은 빠르므로 캐시 불필요할 수 있음
def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """지정된 이름의 tiktoken 인코더를 반환합니다."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except ValueError:
        logging.warning(f"Encoding '{encoding_name}' not found. Using default '{DEFAULT_ENCODING}'.")
        return tiktoken.get_encoding(DEFAULT_ENCODING)

def num_tokens_from_string(string: str, encoding: tiktoken.Encoding) -> int:
    """주어진 문자열의 토큰 수를 반환합니다."""
    if not string:
        return 0
    return len(encoding.encode(string))

def num_tokens_from_messages(messages: List[Dict[str, str]], encoding: tiktoken.Encoding) -> int:
    """메시지 리스트의 총 토큰 수를 계산합니다. OpenAI Cookbook의 계산 방식을 따릅니다."""
    num_tokens = 0
    for message in messages:
        # OpenAI Cookbook에 따른 토큰 계산 방식 (메시지당 4토큰 + 내용 토큰)
        num_tokens += 4
        for key, value in message.items():
            # if value is a string, count its tokens
            if isinstance(value, str):
                 num_tokens += num_tokens_from_string(value, encoding)
            # roles and names also add tokens
            if key == "name":
                num_tokens -= 1 # name이 있으면 role이 생략되어 1 토큰 절약
    # 마지막에 봇의 응답 시작을 나타내는 토큰 추가 (assistant)
    num_tokens += 2 # 모든 응답은 <im_start>assistant<im_sep>으로 시작하는 것으로 추정
    return num_tokens

# allow_output_mutation=True 는 파일 객체와 같은 변경 가능한 객체를 캐시할 때 필요할 수 있음
# 이 함수는 이제 uploaded_file 객체 자체가 아닌, 바이트 내용과 메타데이터를 받습니다.
@st.cache_data(show_spinner=False, hash_funcs={docx.document.Document: id, pd.DataFrame: pd.util.hash_pandas_object})
def read_file(uploaded_file_content_bytes, filename, file_type) -> Tuple[str, Optional[str]]:
    """
    업로드된 파일의 내용을 (bytes) 받아 텍스트 내용을 반환합니다.
    성공 시 (내용, None), 실패 시 ('', 에러 메시지) 반환
    """
    try:
        logging.info(f"Reading file content for: {filename} (Type: {file_type})")
        # BytesIO를 사용하여 파일류 객체처럼 다룹니다.
        file_like_object = io.BytesIO(uploaded_file_content_bytes)

        if file_type == 'text/plain':
            try:
                # BytesIO에서 read() 후 decode
                content = file_like_object.read().decode('utf-8')
            except UnicodeDecodeError:
                logging.warning(f"UTF-8 decoding failed for {filename}, trying cp949.")
                # read()를 다시 호출하면 스트림이 끝에 있을 수 있으므로 seek(0)으로 되돌립니다.
                file_like_object.seek(0)
                content = file_like_object.read().decode('cp949')
            return content, None
        elif file_type == 'application/pdf':
            reader = PdfReader(file_like_object)
            text_parts = []
            for i, page in enumerate(reader.pages):
                try:
                    # extract_text()가 None을 반환하는 경우 빈 문자열 사용
                    page_text = page.extract_text() or ''
                    if page_text: # 빈 문자열이 아니면 추가
                        text_parts.append(page_text)
                except Exception as page_err:
                    logging.warning(f"Error extracting text from page {i+1} of {filename}: {page_err}")
            return '\n'.join(text_parts), None
        elif 'wordprocessingml.document' in file_type:
            doc = docx.Document(file_like_object)
            return '\n'.join(p.text for p in doc.paragraphs), None
        elif 'spreadsheetml.sheet' in file_type:
            df = pd.read_excel(file_like_object, engine='openpyxl')
            return df.to_csv(index=False, sep='\t'), None
        # 이미지 파일 타입은 여기서 텍스트 읽기를 시도하지 않습니다.
        # 이미지 처리는 별도의 로직이 필요하며, 현재는 파일 업로드 후 바로 표시만 합니다.
        elif file_type in ['image/jpeg', 'image/png']:
             # 이미지 파일은 read_file로 텍스트 내용을 추출할 수 없습니다.
             # 이 함수는 텍스트 내용을 반환해야 하므로, 이미지 파일은 여기서 처리하지 않습니다.
             # 이미지 표시 로직은 파일 처리 큐 관리 블록에서 이루어집니다.
             return '', f"이미지 파일은 텍스트로 변환할 수 없습니다: {file_type}"
        else:
            logging.warning(f"Unsupported file type for reading: {file_type}")
            return '', f"지원하지 않는 파일 형식입니다: {file_type}"
    except Exception as e:
        logging.error(f"Error reading file content for {filename}: {e}", exc_info=True)
        return '', f"파일 내용 처리 중 오류 발생: {e}"

@st.cache_data(show_spinner=False)
def summarize_document(text: str, filename: str, model: str, tokenizer: tiktoken.Encoding) -> Tuple[str, Optional[str]]:
    """
    주어진 텍스트를 청크로 나누어 요약하고 결과를 합칩니다.
    성공 시 (요약 내용, None), 실패 시 ('', 에러 메시지) 반환
    """
    if not text:
        return "(문서 내용이 비어 있습니다)", None

    summaries = []
    # 간단한 청크 분할 (토큰 고려 안함, 문자 길이 기준)
    # 더 정교한 토큰 기반 청크 분할이 필요할 수 있습니다.
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    total_chunks = len(chunks)
    logging.info(f"Starting summarization for '{filename}' with {total_chunks} chunks.")

    progress_text = f"'{filename}' 요약 중... (총 {total_chunks}개 청크)"
    progress_bar = st.progress(0, text=progress_text)
    summary_errors = []

    for i, chunk in enumerate(chunks):
        current_progress = (i + 1) / total_chunks
        progress_bar.progress(current_progress, text=f"{progress_text} [{i+1}/{total_chunks}]")

        # 청크 토큰 수 확인 (정교한 토큰 분할이 아니므로 여기서 체크하여 경고만)
        chunk_tokens = num_tokens_from_string(chunk, tokenizer)
        # 요약 모델 컨텍스트 한도 근처면 경고 (간단화 위해 여기서는 MAX_CONTEXT_TOKENS 사용)
        # summarize_document 함수는 모델 정보를 직접 받으므로 MODEL 변수 대신 인자로 받은 model 사용
        if chunk_tokens > MODEL_CONTEXT_LIMITS.get(model, 8192) - 500: # 모델별 실제 한도와 여유 공간 고려
             warning_msg = f"청크 {i+1}의 토큰 수 ({chunk_tokens})가 모델({model}) 한도에 근접하거나 초과할 수 있습니다. 요약이 잘릴 수 있습니다."
             # st.warning(warning_msg) # 너무 많은 경고 방지
             logging.warning(warning_msg)


        try:
            # API 호출
            response = client.chat.completions.create(
                model=model, # 요약에도 동일 모델 사용 (또는 더 저렴한 모델 지정 가능)
                messages=[
                    {'role': 'system', 'content': CHUNK_PROMPT_FOR_SUMMARY},
                    {'role': 'user', 'content': chunk}
                ],
                max_tokens=250, # 요약 길이 제한
                temperature=0.3, # 낮은 온도
                timeout=60 # 타임아웃 설정 (초)
            )
            summary_part = response.choices[0].message.content.strip()
            summaries.append(summary_part)
            sleep(0.15) # API 속도 제한 방지 (약간 증가)
        except Exception as e:
            error_msg = f"청크 {i+1} 요약 중 오류 발생: {e}"
            st.warning(error_msg) # UI에 경고 표시
            logging.error(f"Error summarizing chunk {i+1} of {filename}: {e}", exc_info=True)
            summaries.append(f"(청크 {i+1} 요약 실패)")
            summary_errors.append(error_msg)

    progress_bar.empty() # 진행률 바 숨김
    full_summary = '\n'.join(summaries)
    logging.info(f"Finished summarization for '{filename}'.")
    error_report = "\n".join(summary_errors) if summary_errors else None
    return full_summary, error_report


@st.cache_data(show_spinner=False)
def load_history(path: str) -> List[Dict[str, str]]:
    """JSON 파일에서 대화 기록을 로드합니다."""
    if not os.path.exists(path):
        logging.info(f"History file not found at {path}, starting new history.")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            history = json.load(f)
            logging.info(f"Loaded {len(history)} messages from {path}.")
            return history
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"Error loading or decoding history file {path}: {e}")
        # 손상된 파일 백업 로직은 load_history 외부 호출자가 하도록 하거나,
        # 여기서 파일 존재 시 백업 후 새로 시작하도록 구현 가능
        # 여기서는 간단히 빈 기록 반환
        st.sidebar.warning(f"대화 기록 파일을 읽는 중 오류 발생: {e}. 새 기록으로 시작합니다.")
        return []
    except Exception as e:
        logging.error(f"Error loading history from {path}: {e}", exc_info=True)
        st.sidebar.error(f"대화 기록 로드 중 예상치 못한 오류 발생: {e}")
        return []


def save_history(path: str, msgs: List[Dict[str, str]]):
    """대화 기록을 JSON 파일에 저장합니다."""
    # 시스템 메시지는 저장하지 않습니다.
    msgs_to_save = [msg for msg in msgs if msg['role'] != 'system']
    # 대화 메시지가 없으면 파일 저장 시도 안 함
    if not msgs_to_save:
        if os.path.exists(path):
            # 대화가 비었는데 파일이 있으면 삭제 (초기화 시 외에도 발생 가능)
            try:
                os.remove(path)
                logging.info(f"History file {path} removed as messages are empty.")
            except OSError as e:
                logging.error(f"Failed to remove empty history file {path}: {e}")
        return # 저장할 메시지 없으면 함수 종료

    try:
        # 디렉토리가 없으면 생성 (Streamlit Cloud / 일부 환경 대비)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(msgs_to_save, f, ensure_ascii=False, indent=2)
        # logging.info(f"Saved {len(msgs_to_save)} messages to {path}.") # 너무 자주 로깅될 수 있음
    except Exception as e:
        # Streamlit Cloud에서 쓰기 권한이 없는 경우 에러 발생 가능
        logging.error(f"Error saving history to {path}: {e}", exc_info=True)
        # st.error(f"대화 기록 저장 중 오류 발생: {e}") # 사용자에게 너무 자주 보일 수 있음


# ------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------
# 세션 상태 초기화: 페이지 로드 시 딱 한 번 실행됩니다.
# 'messages' 상태가 존재하지 않으면 (새 세션 또는 초기화 후) history를 로드합니다.
if 'messages' not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = load_history(HISTORY_FILE)
    logging.info(f"Session initialized. Loaded {len(st.session_state.messages)} messages.")

if 'doc_summaries' not in st.session_state:
    st.session_state.doc_summaries: Dict[str, str] = {}
if 'processed_file_ids' not in st.session_state:
    st.session_state.processed_file_ids: set = set()

# 파일 처리 대기열 상태 추가
if 'file_to_summarize' not in st.session_state:
    st.session_state.file_to_summarize: Optional[Dict] = None

# 안전하게 캡처된 파일 정보 저장용 임시 변수 초기화
if 'file_info_to_process_safely_captured' not in st.session_state:
     st.session_state.file_info_to_process_safely_captured: Optional[Dict] = None


# ------------------------------------------------------------------
# SIDEBAR: MODEL, MODE SELECTION & OPTIONS
# ------------------------------------------------------------------
st.sidebar.title("⚙️ 설정")

MODEL = st.sidebar.selectbox(
    '모델 선택',
    list(MODEL_CONTEXT_LIMITS.keys()),
    index=list(MODEL_CONTEXT_LIMITS.keys()).index("gpt-4o") if "gpt-4o" in MODEL_CONTEXT_LIMITS else 0 # 기본값 gpt-4o 시도
)
# MAX_CONTEXT_TOKENS 변수 정의 (MODEL에 따라 달라짐)
MAX_CONTEXT_TOKENS = MODEL_CONTEXT_LIMITS.get(MODEL, 4096)


MODE = st.sidebar.radio('응답 모드', ('Poetic', 'Logical'), index=0, key='mode_selection')


st.sidebar.markdown("---")
st.sidebar.subheader("관리")

# 세션 내용 다운로드 버튼
def build_full_session_content() -> str:
    """문서 요약과 전체 대화 기록을 합쳐 텍스트로 만듭니다."""
    parts = []
    # pd.Timestamp 사용 전에 pandas가 임포트되어 있는지 확인
    # import pandas as pd 가 파일 상단에 있습니다.
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    parts.append(f"Liel Chat Session Content - {timestamp}")
    parts.append(f"Model: {MODEL}, Mode: {MODE}\n")

    if st.session_state.doc_summaries:
        parts.append("===== Uploaded Document Summaries =====")
        # 문서 요약은 최신순이 아닌 파일명 순으로 정렬하여 일관성 유지
        for fname in sorted(st.session_state.doc_summaries.keys()):
            summ = st.session_state.doc_summaries[fname]
            parts.append(f"\n--- Summary: {fname} ---")
            parts.append(summ)
            parts.append("-" * (len(fname) + 16))
        parts.append("\n" + "=" * 30 + "\n")

    parts.append("===== Conversation History =====")
    # 시스템 메시지는 다운로드 내용에 포함하지 않습니다. (save_history와 일관성 유지)
    # 대화 기록은 시간 순서대로 (session_state.messages는 이미 시간 순서)
    msgs_to_include = [msg for msg in st.session_state.messages if msg['role'] != 'system']
    if not msgs_to_include:
         parts.append("(No conversation yet)")
    else:
        for m in msgs_to_include:
            role_icon = "👤 User" if m['role'] == 'user' else "🤖 Liel"
            parts.append(f"\n{role_icon}:\n{m['content']}")
            parts.append("-" * 20)

    return '\n'.join(parts)

# 대화나 문서 요약(시스템 메시지 제외)이 있을 때만 다운로드 버튼 표시
# messages에 시스템 메시지만 있는 경우는 제외
if [msg for msg in st.session_state.messages if msg['role'] != 'system'] or st.session_state.doc_summaries:
    session_content_txt = build_full_session_content()
    download_filename = f"liel_session_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"
    st.sidebar.download_button(
        label="📥 현재 세션 내용 다운로드", # 버튼 레이블 변경
        data=session_content_txt.encode('utf-8'), # UTF-8 인코딩 명시
        file_name=download_filename,
        mime='text/plain',
        help="업로드된 문서 요약과 현재까지의 대화 기록 전체를 텍스트 파일로 다운로드합니다." # 도움말 변경
    )

# 대화 및 문서 요약 초기화 버튼
if st.sidebar.button("🔄 대화 및 문서 요약 초기화"):
    st.session_state.messages = []
    st.session_state.doc_summaries = {}
    st.session_state.processed_file_ids = set()
    st.session_state.file_to_summarize = None # 처리 대기 파일도 초기화
    st.session_state.file_info_to_process_safely_captured = None # 안전 캡처된 정보도 초기화

    # chat_history.json 파일 삭제
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

# 매 스크립트 실행 시 현재 SYSTEM_PROMPT를 메시지 목록의 첫 요소로 관리
# 'messages' 상태가 존재하고 (초기화 후), 첫 요소가 현재 SYSTEM_PROMPT와 다르면 업데이트
# 이 로직은 messages 상태 초기화(load_history) 이후에 실행되어야 합니다.
# 세션 상태 초기화 블록 뒤에 위치합니다.
if 'messages' in st.session_state:
     # 기존 시스템 메시지 제거 (혹시 중복되거나 이전 모드의 시스템 메시지가 있다면)
     st.session_state.messages = [msg for msg in st.session_state.messages if msg['role'] != 'system']
     # 현재 시스템 메시지를 목록 맨 앞에 추가
     st.session_state.messages.insert(0, SYSTEM_PROMPT)
     # save_history 함수에서 시스템 메시지는 저장 시 제외됩니다.


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
    help="텍스트, PDF, 워드, 엑셀 파일은 요약하여 컨텍스트에 포함하고, 이미지 파일은 표시합니다." # 도움말 업데이트
)

# --- File Upload Handling and Queuing: Step 1 - Safely Capture File Info ---
# 이 블록은 Streamlit 재실행 시마다 실행됩니다.
# 파일 업로더 위젯에 새로운 파일 객체가 있는지 확인하고, 필수 정보를 안전하게 캡처합니다.
if uploaded_file is not None:
    try:
        # 파일 객체의 고유 ID, 이름, 타입, 내용(바이트)을 안전하게 가져옵니다.
        # 이 과정에서 AttributeError가 발생할 수 있습니다.
        file_id_now = uploaded_file.id # 이 라인에서 에러가 발생했었습니다.
        file_name_now = uploaded_file.name
        file_type_now = uploaded_file.type
        file_bytes_now = uploaded_file.getvalue() # 파일 내용을 바이트로 즉시 가져옵니다.

        # 정보 캡처 성공 시:
        # 이 파일이 이미 처리되었거나, 이미 처리 대기열에 있거나, 이미 안전하게 캡처되어 있는지 확인합니다.
        is_already_processed = file_id_now in st.session_state.processed_file_ids
        is_already_in_main_queue = ('file_to_summarize' in st.session_state and \
                               st.session_state.file_to_summarize is not None and \
                               st.session_state.file_to_summarize['id'] == file_id_now)
        is_already_safely_captured = ('file_info_to_process_safely_captured' in st.session_state and \
                                      st.session_state.file_info_to_process_safely_captured is not None and \
                                      st.session_state.file_info_to_process_safely_captured['id'] == file_id_now)

        # 새로운 파일이라면 (처리되지 않았고, 큐나 캡처 변수에 없다면):
        if not is_already_processed and not is_already_in_main_queue and not is_already_safely_captured:
            logging.info(f"Detected new file and attempting to safely capture details: {file_name_now} (ID: {file_id_now})")
            # 안전하게 캡처된 정보를 세션 상태 임시 변수에 저장합니다.
            st.session_state.file_info_to_process_safely_captured = {
                'id': file_id_now,
                'name': file_name_now,
                'type': file_type_now,
                'bytes': file_bytes_now # 바이트 내용 저장
            }
            # 정보를 캡처했으니 다음 단계로 이동하기 위해 Streamlit 재실행
            st.rerun()
        # 만약 이미 안전하게 캡처된 파일이라면, 임시 캡처 상태를 비워 다음 캡처를 받을 수 있도록 합니다.
        # (이전 재실행에서 이미 캡처되었지만 아직 하위 단계로 넘어가지 않은 경우)
        # 이 경우는 is_already_safely_captured만 True이고 나머지는 False일 때 해당
        elif is_already_safely_captured and not is_already_processed and not is_already_in_main_queue:
             logging.info(f"File '{file_name_now}' (ID: {file_id_now}) was already safely captured. Clearing capture state.")
             # 임시 캡처 변수 비우기
             st.session_state.file_info_to_process_safely_captured = None
             # 이 경우는 이미 캡처된 정보가 하위 단계로 넘어가서 처리될 것이므로 별도의 rerun은 필요 없습니다.
             pass
        # 이미 처리 완료되었거나 메인 큐에 있는 경우는 아무것도 하지 않습니다.


    except AttributeError as e:
        # uploaded_file이 None이거나 유효하지 않을 때 .id, .name 등에 접근해서 발생하는 에러를 잡습니다.
        # 이는 정상적인 Streamlit 재실행 과정에서 발생할 수 있는 현상입니다.
        logging.warning(f"AttributeError caught during uploaded_file attribute access (likely stale object in rerun): {e}")
        # 이 경우 해당 재실행 주기에서는 파일 정보를 안전하게 캡처하지 않고 건너뜞니다.
        # 유효한 uploaded_file 객체가 있는 다음 재실행 주기에서 다시 시도될 것입니다.
        pass
    except Exception as e:
         # 그 외 예상치 못한 에러를 잡습니다.
         logging.error(f"Unexpected error during uploaded_file attribute access: {e}", exc_info=True)
         pass


# --- File Upload Handling and Queuing: Step 2 - Convert Bytes to Text or Display Image ---
# 이 블록은 안전하게 캡처된 파일 정보(바이트)가 세션 상태에 있을 때 실행됩니다.
# 바이트 내용을 텍스트로 변환하거나 이미지 파일을 표시하고 메인 처리 큐에 추가합니다.
if 'file_info_to_process_safely_captured' in st.session_state and \
   st.session_state.file_info_to_process_safely_captured is not None:

    file_info_captured = st.session_state.file_info_to_process_safely_captured

    # 이 파일 ID가 이미 최종 처리 완료되지 않은 경우에만 진행
    if file_info_captured['id'] not in st.session_state.processed_file_ids:

        logging.info(f"Processing safely captured file info (Bytes to Text/Display) for '{file_info_captured['name']}' (ID: {file_info_captured['id']}).")

        # 안전하게 캡처된 상태를 비워서 이 단계가 한 번만 실행되도록 합니다.
        st.session_state.file_info_to_process_safely_captured = None

        # 이미지 파일인지 확인하고 바로 표시
        if file_info_captured['type'] in ['image/jpeg', 'image/png']:
             logging.info(f"Displaying image file: {file_info_captured['name']}")
             # 이미지 파일은 대화 목록에 메시지 형태로 추가하기보다는 UI에 직접 표시하는 것이 일반적입니다.
             # 여기서는 이미지 파일 자체를 처리 완료로 표시하고 UI에 이미지를 보여줍니다.
             with st.chat_message("system"): # 시스템 메시지 형태로 이미지를 표시할 수도 있습니다.
                 st.image(file_info_captured['bytes'], caption=f"업로드된 이미지: {file_info_captured['name']}", use_column_width=True)

             st.session_state.processed_file_ids.add(file_info_captured['id']) # 이미지 파일 처리 완료로 표시
             st.success(f"🖼️ '{file_info_captured['name']}' 이미지 업로드 및 표시 완료!")
             st.rerun() # UI 업데이트를 위해 재실행

        else: # 텍스트 기반 파일 형식 (txt, pdf, docx, xlsx)
            # read_file 함수를 사용하여 바이트 내용을 텍스트로 변환합니다.
            # read_file expects bytes content, filename, type
            content_text, read_error = read_file(file_info_captured['bytes'], file_info_captured['name'], file_info_captured['type'])

            if read_error:
                st.error(f"'{file_info_captured['name']}' 파일 읽기 실패: {read_error}")
                # 파일 읽기 실패도 처리 완료로 표시하여 무한 루프 방지
                st.session_state.processed_file_ids.add(file_info_captured['id'])
            elif not content_text:
                st.warning(f"'{file_info_captured['name']}' 파일 내용이 비어 있습니다. 요약을 건너뜁니다.")
                st.session_state.processed_file_ids.add(file_info_captured['id']) # 빈 파일도 처리 완료로 표시
            else:
                # 텍스트 내용이 있는 경우 메인 요약 처리 큐에 추가합니다.
                st.session_state.file_to_summarize = {
                    'id': file_info_captured['id'],
                    'name': file_info_captured['name'],
                    'content': content_text # 텍스트 내용 저장
                }
                logging.info(f"File '{file_info_captured['name']}' text content queued for summarization.")
                # 요약 처리 단계로 이동하기 위해 Streamlit 재실행
                st.rerun()


# --- Main Summarization Processing: Step 3 - Summarize Text ---
# 이 블록은 메인 처리 큐(file_to_summarize - 텍스트 내용 포함)에 파일이 있을 때 실행됩니다.
# summarize_document 함수를 호출하여 실제 요약을 수행합니다.
if 'file_to_summarize' in st.session_state and \
   st.session_state.file_to_summarize is not None and \
   st.session_state.file_to_summarize['id'] not in st.session_state.processed_file_ids: # 이미 처리 완료되지 않은 경우

    file_info_to_process = st.session_state.file_to_summarize
    file_id_to_process = file_info_to_process['id']
    filename_to_process = file_info_to_process['name']
    file_content_to_process = file_info_to_process['content'] # 이 시점에서 이미 텍스트 내용

    # 메인 처리 큐 슬롯을 비워서 이 단계가 한 번만 실행되도록 합니다.
    st.session_state.file_to_summarize = None

    logging.info(f"Starting summarization processing from queue: {filename_to_process} (ID: {file_id_to_process})")

    with st.spinner(f"'{filename_to_process}' 처리 및 요약 중..."):
        tokenizer = get_tokenizer()
        # summarize_document 함수는 텍스트 내용을 바로 받습니다.
        # 모델은 사이드바에서 선택된 현재 MODEL 변수를 사용합니다.
        summary, summary_error = summarize_document(file_content_to_process, filename_to_process, MODEL, tokenizer)

        if summary_error:
             st.warning(f"'{filename_to_process}' 요약 중 일부 오류 발생:\n{summary_error}")

        st.session_state.doc_summaries[filename_to_process] = summary
        st.session_state.processed_file_ids.add(file_id_to_process) # 최종 처리 완료 ID 추가

    st.success(f"📄 '{filename_to_process}' 업로드 및 요약 완료!")
    logging.info(f"Successfully processed and summarized file: {filename_to_process}")
    # 요약 완료 후 Streamlit 재실행하여 UI (요약 Expander, 버튼 가시성 등)를 업데이트합니다.
    st.rerun()


# 요약된 문서 표시 (Expander 사용)
if st.session_state.doc_summaries:
    with st.expander("📚 업로드된 문서 요약 보기", expanded=False):
        # 문서 요약은 파일명 순으로 정렬하여 표시
        for fname in sorted(st.session_state.doc_summaries.keys()):
             summ = st.session_state.doc_summaries[fname]
             st.text_area(f"요약: {fname}", summ, height=150, key=f"summary_display_{fname}", disabled=True)
        # 문서 요약만 지우는 버튼
        if st.button("문서 요약만 지우기", key="clear_doc_summaries_btn_exp"):
             st.session_state.doc_summaries = {}
             # 문서 요약만 지우므로 처리 완료된 파일 목록(processed_file_ids)은 그대로 둡니다.
             # 다시 동일 파일을 업로드하면 재처리됩니다.
             # st.session_state.processed_file_ids = set() # 이것은 전체 초기화 시 사용
             st.session_state.file_to_summarize = None # 처리 대기 파일도 초기화
             st.session_state.file_info_to_process_safely_captured = None # 안전 캡처된 정보도 초기화
             logging.info("Document summaries cleared by user from expander button.")
             st.rerun()


# ------------------------------------------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("대화")

# 시스템 메시지는 대화 목록에 표시하지 않습니다.
msgs_to_display = [msg for msg in st.session_state.messages if msg['role'] != 'system']

for message in msgs_to_display:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------------------------------------------
# CHAT INPUT & RESPONSE GENERATION (with Streaming)
# ------------------------------------------------------------------
if prompt := st.chat_input("여기에 메시지를 입력하세요..."):
    # 사용자 메시지 표시 및 기록 (session_state에 추가)
    # 시스템 메시지를 제외한 대화 목록에 추가합니다.
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # 사용자 메시지 즉시 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 컨텍스트 구성 ---
    try:
        tokenizer = get_tokenizer()
        current_model_max_tokens = MODEL_CONTEXT_LIMITS.get(MODEL, 4096)

        # 시스템 프롬프트를 가져옵니다.
        # st.session_state.messages의 첫 요소가 시스템 프롬프트라고 가정합니다.
        # (SYSTEM_PROMPT 정의 및 세션 상태 업데이트 로직에 따라)
        # 메시지 목록이 비어있거나 첫 메시지가 시스템 프롬프트가 아니면 기본 SYSTEM_PROMPT 사용
        current_system_prompt = st.session_state.messages[0] if st.session_state.messages and st.session_state.messages[0]['role'] == 'system' else SYSTEM_PROMPT

        # 시스템 프롬프트 + 예약 토큰을 제외한 실제 사용 가능한 토큰
        base_tokens = num_tokens_from_messages([current_system_prompt], tokenizer)
        available_tokens_for_context = current_model_max_tokens - base_tokens - RESERVED_TOKENS

        if available_tokens_for_context <= 0:
             st.error("설정된 모델의 컨텍스트 길이와 예약 토큰으로 인해 대화 컨텍스트를 구성할 수 없습니다. 모델 변경 또는 예약 토큰 감소를 고려하세요.")
             logging.error("Not enough tokens for context construction after system prompt and reserved tokens.")
             # 오류 발생시켜 하위 로직 중단
             raise ValueError("Context window too small")


        # 실제 API 호출에 사용될 메시지 목록 (시스템 프롬프트 포함)
        conversation_context = [current_system_prompt]
        tokens_used = base_tokens

        # 문서 요약 추가 (토큰 예산 내에서 최신순)
        doc_summary_context = []
        doc_tokens_added = 0
        # st.session_state.doc_summaries.items()는 파일명 순서가 아닐 수 있습니다.
        # 필요하다면 key 또는 timestamp를 저장해서 최신순으로 정렬해야 합니다.
        # 현재는 dictionary 순서를 따르므로, 실제 앱 동작에 따라 최신 업로드 파일 요약이 먼저 추가되지 않을 수 있습니다.
        # 임시로 dictionary를 list로 변환 후 reverse하여 최신순처럼 시뮬레이션
        # 실제 구현 시에는 {filename: {'summary': ..., 'timestamp': ...}} 형태로 저장하는 것이 좋습니다.
        for fname, summ in reversed(list(st.session_state.doc_summaries.items())):
            summary_msg = {'role': 'system', 'content': f"[문서 '{fname}' 요약 참고]\n{summ}"}
            temp_context = [summary_msg]
            summary_tokens = num_tokens_from_messages(temp_context, tokenizer)

            # 사용 가능한 토큰 = 전체 가용 - (현재 사용 + 추가될 요약)
            if available_tokens_for_context - (tokens_used - base_tokens + doc_tokens_added + summary_tokens) >= 0:
                doc_summary_context.insert(0, summary_msg) # 리스트 앞에 추가하여 컨텍스트에서 시스템 프롬프트 뒤에 오도록 함
                doc_tokens_added += summary_tokens
            else:
                logging.warning(f"Document summary '{fname}' skipped due to token limit.")
                break # 더 이상 추가할 수 없으면 중단

        conversation_context.extend(doc_summary_context)
        tokens_used += doc_tokens_added


        # 대화 기록 추가 (토큰 예산 내에서 최신순)
        history_context = []
        history_tokens_added = 0
        # 시스템 메시지를 제외한 대화 기록만 컨텍스트에 추가합니다.
        # st.session_state.messages는 시간 순서대로 쌓이므로 reversed() 사용하면 최신 메시지부터 가져옴
        msgs_to_consider = [msg for msg in st.session_state.messages if msg['role'] != 'system']

        for msg in reversed(msgs_to_consider):
            temp_context = [msg]
            msg_tokens = num_tokens_from_messages(temp_context, tokenizer)

            # 사용 가능한 토큰 = 전체 가용 - (현재 사용(요약포함) - base + 추가될 히스토리)
            if available_tokens_for_context - ((tokens_used - base_tokens) + history_tokens_added + msg_tokens) >= 0:
                history_context.insert(0, msg) # 리스트 앞에 추가하여 시간 순서대로 정렬
                history_tokens_added += msg_tokens
            else:
                logging.warning("Older chat history skipped due to token limit.")
                break # 더 이상 추가할 수 없으면 중단

        conversation_context.extend(history_context)
        tokens_used += history_tokens_added


        # 최종 컨텍스트 토큰 수 로깅
        logging.info(f"Context constructed with {tokens_used} tokens for model {MODEL}.")
        # logging.debug(f"Final conversation context: {conversation_context}") # 필요시 상세 로깅

    except Exception as e:
        # 컨텍스트 구성 중 예상치 못한 오류 발생 시
        st.error(f"대화 컨텍스트 구성 중 오류 발생: {e}")
        logging.error(f"Error constructing conversation context: {e}", exc_info=True)
        # 오류 발생 시 최소한의 컨텍스트만 포함하여 API 호출을 시도하거나 오류 메시지만 출력
        # 시스템 프롬프트와 사용자 메시지만 포함
        current_system_prompt = st.session_state.messages[0] if st.session_state.messages and st.session_state.messages[0]['role'] == 'system' else SYSTEM_PROMPT
        conversation_context = [current_system_prompt, {'role': 'user', 'content': prompt}]
        # 이 시점에서는 tokens_used 계산이 정확하지 않을 수 있습니다.

    # --- API 호출 및 응답 스트리밍 ---
    # 컨텍스트 구성 중 오류가 발생하지 않았거나, 오류 처리 후 최소 컨텍스트가 있는 경우 진행
    # conversation_context가 비어있지 않고 시스템 메시지만 있지 않은 경우
    if conversation_context and any(msg['role'] != 'system' for msg in conversation_context):
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # 응답 표시 영역
            full_response = ""
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=conversation_context,
                    stream=True,
                    temperature=0.75 if MODE == 'Poetic' else 0.4, # 모드별 온도 조절
                    # max_tokens= # 필요시 최대 응답 길이 설정 가능
                    timeout=120 # 스트리밍 타임아웃 (초)
                )
                # 스트림 처리 및 표시
                for chunk in stream:
                    # 델타 내용이 None이 아닌 경우에만 처리
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                        chunk_content = chunk.choices[0].delta.content
                        full_response += chunk_content
                        # 응답이 너무 길어지면 UI 업데이트 성능 저하될 수 있으므로 적당한 단위로 업데이트
                        message_placeholder.markdown(full_response + "▌") # 커서 효과

                # 스트림이 끝난 후 최종 응답 표시
                message_placeholder.markdown(full_response)
                logging.info(f"Assistant response received (length: {len(full_response)} chars).")

            except Exception as e:
                # API 호출 또는 스트리밍 중 오류 발생 시
                full_response = f"⚠️ 죄송합니다, 응답 생성 중 오류가 발생했습니다: {e}"
                message_placeholder.error(full_response)
                logging.error(f"Error during OpenAI API call or streaming: {e}", exc_info=True)

    else:
         # 컨텍스트 구성이 실패했거나 유효한 메시지가 없어 API 호출을 건너뛴 경우
         full_response = "⚠️ 대화 컨텍스트 구성 실패로 응답을 생성할 수 없습니다. 오류 로그를 확인하세요."
         st.chat_message("assistant").error(full_response)


    # 응답 기록 저장 (시스템 메시지 제외)
    # 정상 응답이나 오류 메시지 모두 st.session_state.messages에는 추가합니다.
    # save_history 함수에서 시스템 메시지나 특정 오류 메시지(필요시)를 제외하도록 처리합니다.
    if full_response:
         st.session_state.messages.append({'role': 'assistant', 'content': full_response})
         # save_history 함수는 시스템 메시지나 오류 메시지를 저장하지 않도록 이미 구현되어 있습니다.
         save_history(HISTORY_FILE, st.session_state.messages)


# --- Footer or additional info ---
st.sidebar.markdown("---")
st.sidebar.caption("Liel Chatbot v1.5") # 버전 업데이트
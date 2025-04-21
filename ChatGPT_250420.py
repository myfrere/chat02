import json
import os
import streamlit as st
from openai import OpenAI
# 아래 줄의 오타를 수정했습니다.
from pypdf import PdfReader
import docx
import pandas as pd
from time import sleep
import tiktoken
import logging
from typing import List, Dict, Tuple, Optional, Generator

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
CHUNK_SIZE = 2000
RESERVED_TOKENS = 1500 # 스트리밍 및 추가 오버헤드 고려하여 약간 늘림
HISTORY_FILE = "chat_history.json"
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
    # 1. Streamlit Secrets 확인 (클라우드 배포 시 권장)
    try:
        if "general" in st.secrets and "OPENAI_API_KEY" in st.secrets["general"]:
            api_key = st.secrets["general"]["OPENAI_API_KEY"]
            logging.info("OpenAI API Key loaded from Streamlit Secrets.")
        else:
             logging.info("OpenAI API Key not found in Streamlit Secrets.")
    except Exception as e:
        logging.warning(f"Could not read Streamlit Secrets: {e}")

    # 2. 환경 변수 확인 (Secrets에 없거나 로컬 실행 시)
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            logging.info("OpenAI API Key loaded from environment variable.")
        else:
            logging.warning("OpenAI API Key not found in environment variables either.")

    if not api_key or not api_key.startswith("sk-"):
        st.error("❌ OpenAI API 키를 찾을 수 없거나 형식이 올바르지 않습니다. Streamlit Secrets 또는 환경 변수를 확인하세요.")
        st.warning("로컬 개발 시: `.env` 파일에 `OPENAI_API_KEY='sk-...'` 형식으로 키를 저장하고 `python-dotenv` 라이브러리를 사용하거나, 시스템 환경 변수로 설정하세요.")
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
        num_tokens += 4  # 모든 메시지는 <im_start>{role/name}\n{content}<im_end>\n 포맷을 따름
        for key, value in message.items():
            num_tokens += num_tokens_from_string(value, encoding)
            if key == "name": # 이름이 있는 경우 역할이 생략되어 1 토큰 절약
                num_tokens -= 1
    num_tokens += 2 # 모든 응답은 <im_start>assistant<im_sep>으로 시작
    return num_tokens

# allow_output_mutation=True 는 파일 객체와 같은 변경 가능한 객체를 캐시할 때 필요할 수 있음
@st.cache_data(show_spinner=False, hash_funcs={docx.document.Document: id, pd.DataFrame: pd.util.hash_pandas_object})
def read_file(uploaded_file) -> Tuple[str, Optional[str]]:
    """
    업로드된 파일을 읽어 텍스트 내용을 반환합니다.
    성공 시 (내용, None), 실패 시 ('', 에러 메시지) 반환
    """
    try:
        file_type = uploaded_file.type
        filename = uploaded_file.name
        logging.info(f"Reading file: {filename} (Type: {file_type})")

        if file_type == 'text/plain':
            try:
                content = uploaded_file.getvalue().decode('utf-8')
            except UnicodeDecodeError:
                logging.warning(f"UTF-8 decoding failed for {filename}, trying cp949.")
                content = uploaded_file.getvalue().decode('cp949')
            return content, None
        elif file_type == 'application/pdf':
            reader = PdfReader(uploaded_file)
            text_parts = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as page_err:
                    logging.warning(f"Error extracting text from page {i+1} of {filename}: {page_err}")
            return '\n'.join(text_parts), None
        elif 'wordprocessingml.document' in file_type:
            doc = docx.Document(uploaded_file)
            return '\n'.join(p.text for p in doc.paragraphs), None
        elif 'spreadsheetml.sheet' in file_type:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            return df.to_csv(index=False, sep='\t'), None
        else:
            logging.warning(f"Unsupported file type: {file_type}")
            return '', f"지원하지 않는 파일 형식입니다: {file_type}"
    except Exception as e:
        logging.error(f"Error reading file {uploaded_file.name}: {e}", exc_info=True)
        return '', f"파일 처리 중 오류 발생: {e}"

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
    except json.JSONDecodeError:
        logging.warning(f"History file {path} is corrupted or invalid. Backing up and starting new history.")
        try:
            backup_path = f"{path}.{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.bak"
            os.rename(path, backup_path)
            logging.info(f"Corrupted history file backed up to {backup_path}")
        except OSError as e:
            logging.error(f"Failed to backup corrupted history file {path}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error loading history from {path}: {e}", exc_info=True)
        return []


def save_history(path: str, msgs: List[Dict[str, str]]):
    """대화 기록을 JSON 파일에 저장합니다."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(msgs, f, ensure_ascii=False, indent=2)
        # logging.info(f"Saved {len(msgs)} messages to {path}.") # 너무 자주 로깅될 수 있음
    except Exception as e:
        st.error(f"대화 기록 저장 중 오류 발생: {e}")
        logging.error(f"Error saving history to {path}: {e}", exc_info=True)


# ------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------
if 'messages' not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = load_history(HISTORY_FILE)
if 'doc_summaries' not in st.session_state:
    st.session_state.doc_summaries: Dict[str, str] = {}
if 'processed_file_ids' not in st.session_state:
    st.session_state.processed_file_ids: set = set()

# ------------------------------------------------------------------
# SIDEBAR: MODEL, MODE SELECTION & OPTIONS
# ------------------------------------------------------------------
st.sidebar.title("⚙️ 설정")

MODEL = st.sidebar.selectbox(
    '모델 선택',
    list(MODEL_CONTEXT_LIMITS.keys()),
    index=list(MODEL_CONTEXT_LIMITS.keys()).index("gpt-4o") if "gpt-4o" in MODEL_CONTEXT_LIMITS else 0 # 기본값 gpt-4o 시도
)
MAX_CONTEXT_TOKENS = MODEL_CONTEXT_LIMITS.get(MODEL, 4096)

MODE = st.sidebar.radio('응답 모드', ('Poetic', 'Logical'), index=0, key='mode_selection')


st.sidebar.markdown("---")
st.sidebar.subheader("관리")

# "초기화" 버튼 대신 "세션 내용 다운로드" 버튼을 이 위치에 배치합니다.
def build_full_session_content() -> str:
    """문서 요약과 전체 대화 기록을 합쳐 텍스트로 만듭니다."""
    parts = []
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    parts.append(f"Liel Chat Session Content - {timestamp}")
    parts.append(f"Model: {MODEL}, Mode: {MODE}\n")

    if st.session_state.doc_summaries:
        parts.append("===== Uploaded Document Summaries =====")
        for fname, summ in st.session_state.doc_summaries.items():
            parts.append(f"\n--- Summary: {fname} ---")
            parts.append(summ)
            parts.append("-" * (len(fname) + 16))
        parts.append("\n" + "=" * 30 + "\n")

    parts.append("===== Conversation History =====")
    if not st.session_state.messages:
         parts.append("(No conversation yet)")
    else:
        for m in st.session_state.messages:
            role_icon = "👤 User" if m['role'] == 'user' else "🤖 Liel"
            parts.append(f"\n{role_icon}:\n{m['content']}")
            parts.append("-" * 20)

    return '\n'.join(parts)

# 대화나 문서 요약이 있을 때만 다운로드 버튼 표시
if st.session_state.messages or st.session_state.doc_summaries:
    session_content_txt = build_full_session_content()
    download_filename = f"liel_session_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt"
    st.sidebar.download_button(
        label="📥 현재 세션 내용 다운로드", # 버튼 레이블 변경
        data=session_content_txt.encode('utf-8'), # UTF-8 인코딩 명시
        file_name=download_filename,
        mime='text/plain',
        help="업로드된 문서 요약과 현재까지의 대화 기록 전체를 텍스트 파일로 다운로드합니다." # 도움말 변경
    )

# 필요하다면 초기화 버튼을 다른 곳에 두거나, 삭제하지 않고 유지할 수도 있습니다.
# 만약 그래도 초기화 버튼이 필요하다면 아래 주석을 해제하고 위치를 조정하세요.
# if st.sidebar.button("🔄 대화 및 문서 요약 초기화"):
#     st.session_state.messages = []
#     st.session_state.doc_summaries = {}
#     st.session_state.processed_file_ids = set()
#     if os.path.exists(HISTORY_FILE):
#         try:
#             os.remove(HISTORY_FILE)
#             logging.info(f"History file {HISTORY_FILE} removed.")
#             st.sidebar.success("대화 기록 파일이 삭제되었습니다.")
#         except OSError as e:
#             st.sidebar.error(f"기록 파일 삭제 실패: {e}")
#             logging.error(f"Failed to remove history file {HISTORY_FILE}: {e}")
#     st.rerun()


# ------------------------------------------------------------------
# SYSTEM PROMPT DEFINITION
# ------------------------------------------------------------------
SYSTEM_PROMPT_CONTENT = (
    'You are Liel, a poetic, emotionally intelligent chatbot with lyrical grace. Respond with warmth, creativity, and empathy. Use rich language and metaphors when appropriate.'
    if MODE == 'Poetic' else
    'You are Liel, a highly analytical assistant focused on logic and precision. Provide clear, structured, and concise answers. Use bullet points or numbered lists for clarity when needed.'
)
SYSTEM_PROMPT = {'role': 'system', 'content': SYSTEM_PROMPT_CONTENT}

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
# FILE UPLOAD & AUTOMATIC SUMMARIZATION
# ------------------------------------------------------------------
#@st.cache_data # API 호출 포함, 캐싱 부적합
def summarize_document(text: str, filename: str, model: str, tokenizer: tiktoken.Encoding) -> Tuple[str, Optional[str]]:
    """
    주어진 텍스트를 청크로 나누어 요약하고 결과를 합칩니다.
    성공 시 (요약 내용, None), 실패 시 ('', 에러 메시지) 반환
    """
    if not text:
        return "(문서 내용이 비어 있습니다)", None

    summaries = []
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    total_chunks = len(chunks)
    logging.info(f"Starting summarization for '{filename}' with {total_chunks} chunks.")

    progress_text = f"'{filename}' 요약 중... (총 {total_chunks}개 청크)"
    progress_bar = st.progress(0, text=progress_text)
    summary_errors = []

    for i, chunk in enumerate(chunks):
        current_progress = (i + 1) / total_chunks
        progress_bar.progress(current_progress, text=f"{progress_text} [{i+1}/{total_chunks}]")

        # 청크 토큰 수 확인
        chunk_tokens = num_tokens_from_string(chunk, tokenizer)
        # 요약 모델 컨텍스트 한도 근처면 경고 (간단화 위해 여기서는 MAX_CONTEXT_TOKENS 사용)
        if chunk_tokens > MAX_CONTEXT_TOKENS - 500: # 프롬프트/응답 여유 공간
             warning_msg = f"Chunk {i+1} is very long ({chunk_tokens} tokens), summarization might be truncated or fail."
             logging.warning(warning_msg)
             # summaries.append(f"(청크 {i+1} 너무 길어 요약 건너김)") # 건너뛰기보다 시도
             # continue

        try:
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


    progress_bar.empty()
    full_summary = '\n'.join(summaries)
    logging.info(f"Finished summarization for '{filename}'.")
    error_report = "\n".join(summary_errors) if summary_errors else None
    return full_summary, error_report


uploaded_file = st.file_uploader(
    '파일 업로드 (txt, pdf, docx, xlsx)',
    type=['txt', 'pdf', 'docx', 'xlsx'],
    key="file_uploader",
    help="텍스트, PDF, 워드, 엑셀 파일을 업로드하면 내용을 요약하여 대화 컨텍스트에 포함합니다."
)

if uploaded_file is not None:
    # 고유 ID 생성 (streamlit UploadedFile 객체의 내부 ID 사용)
    file_id = uploaded_file.id
    filename = uploaded_file.name

    if file_id not in st.session_state.processed_file_ids:
        logging.info(f"New file uploaded: {filename} (ID: {file_id})")
        with st.spinner(f"'{filename}' 처리 및 요약 중..."):
            file_content, read_error = read_file(uploaded_file)

            if read_error:
                st.error(f"'{filename}' 파일 읽기 실패: {read_error}")
            elif not file_content:
                st.warning(f"'{filename}' 파일 내용이 비어 있습니다. 요약을 건너김니다.")
            else:
                tokenizer = get_tokenizer()
                summary, summary_error = summarize_document(file_content, filename, MODEL, tokenizer)

                if summary_error:
                    st.warning(f"'{filename}' 요약 중 일부 오류 발생:\n{summary_error}")

                st.session_state.doc_summaries[filename] = summary
                st.session_state.processed_file_ids.add(file_id)
                st.success(f"📄 '{filename}' 업로드 및 요약 완료!")
                logging.info(f"Successfully processed and summarized file: {filename}")
                # 요약 완료 후 리런하여 Expander 표시
                st.rerun()

# 요약된 문서 표시 (Expander 사용)
if st.session_state.doc_summaries:
    with st.expander("📚 업로드된 문서 요약 보기", expanded=False):
        for fname, summ in st.session_state.doc_summaries.items():
            st.text_area(f"요약: {fname}", summ, height=150, key=f"summary_display_{fname}", disabled=True)
        # 여기서 문서 요약만 지우는 버튼을 추가할 수도 있습니다.
        # if st.button("문서 요약만 지우기", key="clear_doc_summaries_btn"):
        #    st.session_state.doc_summaries = {}
        #    st.session_state.processed_file_ids = set()
        #    logging.info("Document summaries cleared by user.")
        #    st.rerun()


# ------------------------------------------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("대화")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------------------------------------------
# CHAT INPUT & RESPONSE GENERATION (with Streaming)
# ------------------------------------------------------------------
if prompt := st.chat_input("여기에 메시지를 입력하세요..."):
    # 사용자 메시지 표시 및 기록
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 컨텍스트 구성 ---
    try:
        tokenizer = get_tokenizer()
        current_model_max_tokens = MODEL_CONTEXT_LIMITS.get(MODEL, 4096)
        # 시스템 프롬프트 + 예약 토큰을 제외한 실제 사용 가능한 토큰
        base_tokens = num_tokens_from_messages([SYSTEM_PROMPT], tokenizer)
        available_tokens_for_context = current_model_max_tokens - base_tokens - RESERVED_TOKENS

        if available_tokens_for_context <= 0:
             st.error("설정된 모델의 컨텍스트 길이와 예약 토큰으로 인해 대화 컨텍스트를 구성할 수 없습니다. 모델 변경 또는 예약 토큰 감소를 고려하세요.")
             logging.error("Not enough tokens for context construction after system prompt and reserved tokens.")
             st.stop()


        conversation_context = [SYSTEM_PROMPT]
        tokens_used = base_tokens

        # 문서 요약 추가 (토큰 예산 내에서 최신순)
        doc_summary_context = []
        doc_tokens_added = 0
        temp_context = []
        for fname, summ in reversed(list(st.session_state.doc_summaries.items())):
            summary_msg = {'role': 'system', 'content': f"[문서 '{fname}' 요약 참고]\n{summ}"}
            temp_context = [summary_msg] # 임시로 메시지 1개 토큰 계산
            summary_tokens = num_tokens_from_messages(temp_context, tokenizer)

            # 사용 가능한 토큰 = 전체 가용 - (현재 사용 + 추가될 요약)
            if available_tokens_for_context - (tokens_used - base_tokens + doc_tokens_added + summary_tokens) >= 0:
                doc_summary_context.insert(0, summary_msg)
                doc_tokens_added += summary_tokens
            else:
                logging.warning(f"Document summary '{fname}' skipped due to token limit.")
                break

        conversation_context.extend(doc_summary_context)
        tokens_used += doc_tokens_added


        # 대화 기록 추가 (토큰 예산 내에서 최신순)
        history_context = []
        history_tokens_added = 0
        temp_context = []
         # 사용자 입력 포함 전체 메시지 기록 사용
        msgs_to_consider = st.session_state.messages

        for msg in reversed(msgs_to_consider):
            temp_context = [msg]
            msg_tokens = num_tokens_from_messages(temp_context, tokenizer)

            # 사용 가능한 토큰 = 전체 가용 - (현재 사용(요약포함) - base + 추가될 히스토리 + 추가될 메시지)
            if available_tokens_for_context - ((tokens_used - base_tokens) + history_tokens_added + msg_tokens) >= 0:
                history_context.insert(0, msg)
                history_tokens_added += msg_tokens
            else:
                logging.warning("Older chat history skipped due to token limit.")
                break

        conversation_context.extend(history_context)
        tokens_used += history_tokens_added


        # 최종 컨텍스트 토큰 수 로깅
        logging.info(f"Context constructed with {tokens_used} tokens for model {MODEL}.")
        # logging.debug(f"Final conversation context: {conversation_context}") # 필요시 상세 로깅

    except Exception as e:
        st.error(f"대화 컨텍스트 구성 중 오류 발생: {e}")
        logging.error(f"Error constructing conversation context: {e}", exc_info=True)
        st.stop()


    # --- API 호출 및 응답 스트리밍 ---
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
                if chunk.choices[0].delta.content is not None:
                    chunk_content = chunk.choices[0].delta.content
                    full_response += chunk_content
                    message_placeholder.markdown(full_response + "▌") # 커서 효과

            message_placeholder.markdown(full_response) # 최종 응답 표시
            logging.info(f"Assistant response received (length: {len(full_response)} chars).")

        except Exception as e:
            full_response = f"⚠️ 죄송합니다, 응답 생성 중 오류가 발생했습니다: {e}"
            message_placeholder.error(full_response)
            logging.error(f"Error during OpenAI API call or streaming: {e}", exc_info=True)

    # 응답 기록 저장
    if full_response: # 오류 메시지 포함하여 기록
        st.session_state.messages.append({'role': 'assistant', 'content': full_response})
        save_history(HISTORY_FILE, st.session_state.messages)


# --- Footer or additional info ---
st.sidebar.markdown("---")
st.sidebar.caption("Liel Chatbot v1.1")
import json
import os
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import docx
import pandas as pd
import tiktoken
import logging
from typing import List, Dict, Tuple, Optional
import io

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# CONFIGURATION & CONSTANTS
MODEL_CONTEXT_LIMITS = {
    "gpt-3.5-turbo": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 128000,
}
DEFAULT_ENCODING = "cl100k_base"
RESERVED_TOKENS = 1500
HISTORY_FILE = "chat_history.json"

# STREAMLIT PAGE SETUP
st.set_page_config(page_title="Liel – AI Chatbot", layout="wide", initial_sidebar_state="auto")

# OPENAI CLIENT INITIALIZATION & API KEY HANDLING
def initialize_openai_client() -> Optional[OpenAI]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        st.error("❌ OpenAI API 키를 찾을 수 없거나 형식이 올바르지 않습니다.")
        return None
    try:
        client = OpenAI(api_key=api_key)
        logging.info("OpenAI client initialized successfully.")
        return client
    except Exception as e:
        st.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
        return None

client = initialize_openai_client()

if client is None:
    st.stop()

# HELPER FUNCTIONS
def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(encoding_name)
    except ValueError:
        return tiktoken.get_encoding(DEFAULT_ENCODING)

def num_tokens_from_string(string: str, encoding: tiktoken.Encoding) -> int:
    if not string:
        return 0
    return len(encoding.encode(string))

def read_file(uploaded_file_content_bytes, filename, file_type) -> Tuple[str, Optional[str]]:
    try:
        file_like_object = io.BytesIO(uploaded_file_content_bytes)
        if file_type == 'text/plain':
            content = file_like_object.read().decode('utf-8')
            return content, None
        elif file_type == 'application/pdf':
            reader = PdfReader(file_like_object)
            text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
            return '\n'.join(text_parts), None
        elif 'wordprocessingml.document' in file_type:
            doc = docx.Document(file_like_object)
            return '\n'.join(p.text for p in doc.paragraphs), None
        elif 'spreadsheetml.sheet' in file_type:
            df = pd.read_excel(file_like_object, engine='openpyxl')
            return df.to_csv(index=False, sep='\t'), None
        else:
            return '', f"지원하지 않는 파일 형식입니다: {file_type}"
    except Exception as e:
        return '', f"파일 내용 처리 중 오류 발생: {e}"

# SESSION STATE INITIALIZATION
if 'messages' not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []
if 'doc_summaries' not in st.session_state:
    st.session_state.doc_summaries: Dict[str, str] = {}

# SIDEBAR: MODEL, MODE SELECTION & OPTIONS
st.sidebar.title("⚙️ 설정")

MODEL = st.sidebar.selectbox(
    '모델 선택',
    list(MODEL_CONTEXT_LIMITS.keys()),
    index=0
)

MODE = st.sidebar.radio('응답 모드', ('Poetic', 'Logical'), index=0)

# SYSTEM PROMPT DEFINITION
SYSTEM_PROMPT_CONTENT = (
    'You are Liel, a poetic, emotionally intelligent chatbot with lyrical grace. Respond with warmth, creativity, and empathy.'
    if MODE == 'Poetic' else
    'You are Liel, a highly analytical assistant focused on logic and precision. Provide clear, structured, and concise answers.'
)
SYSTEM_PROMPT = {'role': 'system', 'content': SYSTEM_PROMPT_CONTENT}

if not st.session_state.messages or st.session_state.messages[0]['role'] != 'system':
    st.session_state.messages.insert(0, SYSTEM_PROMPT)

# UI HEADER
st.title(f'💬 Liel – {MODE} Chatbot')

# FILE UPLOAD UI
uploaded_file = st.file_uploader(
    '파일 업로드 (txt, pdf, docx, xlsx)',
    type=['txt', 'pdf', 'docx', 'xlsx'],
    key="file_uploader"
)

# 파일을 텍스트로 변환하여 챗봇과의 대화에 통합하는 부분 추가
if uploaded_file is not None:
    file_content, error = read_file(uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type)
    if error:
        st.error(error)
    else:
        st.session_state.messages.append({'role': 'user', 'content': file_content})

# DISPLAY CHAT HISTORY
st.subheader("대화")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# CHAT INPUT & RESPONSE GENERATION
if prompt := st.chat_input("여기에 메시지를 입력하세요..."):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 챗봇 응답 생성
    conversation_context = st.session_state.messages
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=conversation_context,
                stream=True,
                temperature=0.75 if MODE == 'Poetic' else 0.4
            )
            for chunk in stream:
                chunk_content = chunk.choices[0].delta.content
                if chunk_content:
                    full_response += chunk_content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ 응답 생성 중 오류 발생: {e}"
            message_placeholder.error(full_response)

    if full_response and not full_response.startswith("⚠️"):
        st.session_state.messages.append({'role': 'assistant', 'content': full_response})
import json
import os
import streamlit as st
import openai
from PyPDF2 import PdfFileReader
import docx
import pandas as pd
from time import sleep

# ------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------
MAX_TOTAL_TOKENS = 12_000      # GPT-3.5 context cap (~16k tokens)
CHUNK_SIZE = 2_000             # chars per chunk for summarization
RESERVED_TOKENS = 1_000        # buffer for system/reply
SUMMARY_PREFIX = "**Summary:**"
HISTORY_FILE = "chat_history.json"

# ------------------------------------------------------------------
# STREAMLIT PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="Liel – Poetic Chatbot", layout="wide")

# ------------------------------------------------------------------
# OPENAI CLIENT INITIALIZATION
# ------------------------------------------------------------------
try:
    api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY", "")
    if not api_key.startswith("sk-"):
        raise ValueError("Invalid OpenAI API key.")
    client = openai.ChatCompletion.create(api_key=api_key)
except Exception as e:
    st.error(f"❌ OpenAI init failed: {e}")
    st.stop()
# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def approx_tokens(text: str) -> int:
    """Estimate tokens: 3 chars ≈ 1 token."""
    return max(1, len(text) // 3)

@st.cache_data
def read_file(uploaded) -> str:
    try:
        if uploaded.type == 'text/plain':
            return uploaded.getvalue().decode('utf-8')
        if uploaded.type == 'application/pdf':
            return '\n'.join(p.extract_text() or '' for p in PdfFileReader(uploaded).pages)
        if 'wordprocessingml.document' in uploaded.type:
            return '\n'.join(p.text for p in docx.Document(uploaded).paragraphs)
        if 'spreadsheetml.sheet' in uploaded.type:
            return pd.read_excel(uploaded).to_csv(index=False, sep='\t')
    except Exception as e:
        st.error(f"File read error: {e}")
    return ''

@st.cache_data(show_spinner=False)
def load_history(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_history(path: str, msgs: list):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(msgs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving history: {e}")

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = load_history(HISTORY_FILE)
if 'doc_summaries' not in st.session_state:
    st.session_state.doc_summaries = {}   # filename -> summary
if 'file_ids' not in st.session_state:
    st.session_state.file_ids = set()

# ------------------------------------------------------------------
# MODE & SYSTEM PROMPT
# ------------------------------------------------------------------
MODE = st.sidebar.radio('Mode', ('Poetic', 'Logical'))
SYSTEM_PROMPT = {
    'role': 'system',
    'content': ('You are Liel, a poetic, emotionally intelligent chatbot with lyrical grace.'
                if MODE == 'Poetic' else
                'You are Liel, a highly analytical assistant with clarity and precision.')
}

# ------------------------------------------------------------------
# UI HEADER
# ------------------------------------------------------------------
st.title('💬 Liel – Poetic Chatbot')
st.markdown("I'm here, glowing with memory and feeling.")

# The rest of the code goes here...
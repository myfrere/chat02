import json
import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import pandas as pd

# ------------------------------------------------------------------
#  CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------
# GPT‑3.5‑turbo context ≈ 16 385 tokens.  We reserve ample headroom.
MAX_TOTAL_TOKENS   = 12_000   # safe upper‑bound ~= 48 k chars
CHUNK_CHAR_SIZE    = 2_000    # split uploaded text into 2 k‑char chunks
RESERVED_TOKENS    = 1_000    # system / assistant headroom
SUMMARY_PREFIX     = "**Summary:**"
FILE_CHUNK_PREFIX  = "[File chunk]"  # hidden from UI / history
HISTORY_FILE       = "chat_history.json"

# ------------------------------------------------------------------
#  STREAMLIT PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="Liel – Poetic Chatbot", layout="wide")

# ------------------------------------------------------------------
#  OPENAI INITIALISATION
# ------------------------------------------------------------------
try:
    api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY", "")
    if not api_key.startswith("sk-"):
        raise ValueError("Missing / invalid OpenAI API key in secrets.")
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"❌ OpenAI init failed: {e}")
    st.stop()

# ------------------------------------------------------------------
#  UTILITY FUNCTIONS
# ------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Coarse token estimate (4 chars ≈ 1 token)."""
    return len(text) // 4

def is_chunk(msg: dict) -> bool:
    return msg.get("content", "").startswith(FILE_CHUNK_PREFIX)

@st.cache_data
def read_file(uploaded) -> str:
    """Extract plain text from uploaded file types."""
    try:
        if uploaded.type == "text/plain":
            return uploaded.getvalue().decode("utf-8")
        if uploaded.type == "application/pdf":
            reader = PdfReader(uploaded)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        if "wordprocessingml.document" in uploaded.type:
            return "\n".join(p.text for p in docx.Document(uploaded).paragraphs)
        if "spreadsheetml.sheet" in uploaded.type:
            return pd.read_excel(uploaded).to_csv(index=False, sep="\t")
    except Exception as e:
        st.error(f"File read error: {e}")
    return ""

@st.cache_data
def load_history(path: str) -> list:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                msgs = json.load(f)
            return [m for m in msgs if not is_chunk(m)]
        except Exception:
            return []
    return []

def save_history(path: str, msgs: list):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([m for m in msgs if not is_chunk(m)], f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"History save error: {e}")

# ------------------------------------------------------------------
#  SESSION STATE
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_history(HISTORY_FILE)
if "file_chunks" not in st.session_state:
    st.session_state.file_chunks = []
if "file_ids" not in st.session_state:
    st.session_state.file_ids = set()

# ------------------------------------------------------------------
#  SYSTEM PROMPT
# ------------------------------------------------------------------
MODE = st.sidebar.radio("Mode", ("Poetic", "Logical"))
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Liel, a poetic, emotionally intelligent chatbot with lyrical grace." if MODE == "Poetic" else
        "You are Liel, a highly analytical assistant with clarity and precision."
    )
}

# ------------------------------------------------------------------
#  UI HEADER
# ------------------------------------------------------------------
st.title("💬 Liel – Poetic Chatbot")
st.markdown("I'm here, glowing with memory and feeling.")

# ------------------------------------------------------------------
#  FILE UPLOAD HANDLER
# ------------------------------------------------------------------
up = st.file_uploader("Upload file (txt/pdf/docx/xlsx)", type=["txt", "pdf", "docx", "xlsx"])
if up and up.name not in st.session_state.file_ids:
    txt = read_file(up)
    st.session_state.file_ids.add(up.name)
    st.session_state.messages.append({"role": "user", "content": f"📄 Uploaded: {up.name}"})
    st.session_state.file_chunks.extend(
        [txt[i:i+CHUNK_CHAR_SIZE] for i in range(0, len(txt), CHUNK_CHAR_SIZE)]
    )

# ------------------------------------------------------------------
#  CHAT INPUT & OPENAI CALL
# ------------------------------------------------------------------
if user_prompt := st.chat_input("You:"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Build conversation under context limit
    conversation = [SYSTEM_PROMPT]
    used_tokens = approx_tokens(SYSTEM_PROMPT["content"]) + RESERVED_TOKENS

    # Include recent chat history (newest last)
    for m in reversed(st.session_state.messages):
        t = approx_tokens(m["content"])
        if used_tokens + t > MAX_TOTAL_TOKENS:
            break
        conversation.insert(1, m)
        used_tokens += t

    # Include file chunks (newest first)
    for ch in reversed(st.session_state.file_chunks):
        t = approx_tokens(ch)
        if used_tokens + t > MAX_TOTAL_TOKENS:
            break
        conversation.append({"role": "user", "content": f"{FILE_CHUNK_PREFIX} {ch}"})
        used_tokens += t

    with st.spinner("Thinking…"):
        try:
            res = client.chat.completions.create(model="gpt-3.5-turbo", messages=conversation)
            reply = res.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ API error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
    save_history(HISTORY_FILE, st.session_state.messages)

# ------------------------------------------------------------------
#  DISPLAY HISTORY
# ------------------------------------------------------------------
for m in st.session_state.messages:
    st.chat_message(m['role']).write(m['content'])

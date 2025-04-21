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
MAX_TOTAL_TOKENS   = 80_000   # hard safety‑cap (≈ 320 k chars)
CHUNK_CHAR_SIZE    = 5_000    # split uploaded text into 5 k‑char chunks
RESERVED_TOKENS    = 4_000    # 헤더·시스템프롬프트·모델 답변 등 예비치
SUMMARY_PREFIX     = "**Summary:**"
FILE_CHUNK_PREFIX  = "[File chunk]"   # marker hidden from UI / history
HISTORY_FILE       = "chat_history.json"

# ------------------------------------------------------------------
#  STREAMLIT PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="Liel – Poetic Chatbot", layout="wide")

# ------------------------------------------------------------------
#  OPENAI CLIENT INIT
# ------------------------------------------------------------------
try:
    api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY", "")
    if not api_key.startswith("sk-"):
        raise ValueError("OpenAI API key missing or invalid.")
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"❌ OpenAI init failed: {e}")
    st.stop()

# ------------------------------------------------------------------
#  UTILITY FUNCTIONS
# ------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Very safe upper‑bound: 1 token ≈ 4 chars."""
    return len(text) // 4

def is_file_chunk(msg: dict) -> bool:
    return msg.get("content", "").startswith(FILE_CHUNK_PREFIX)

@st.cache_data
def read_file(uploaded) -> str:
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
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            msgs = json.load(f)
        return [m for m in msgs if not is_file_chunk(m)]
    except Exception:
        return []

def save_history(path: str, msgs: list):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([m for m in msgs if not is_file_chunk(m)], f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"History save error: {e}")

# ------------------------------------------------------------------
#  SESSION STATE INITIALISE
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_history(HISTORY_FILE)
if "file_chunks" not in st.session_state:
    st.session_state.file_chunks = []   # raw text chunks for model only
if "file_ids" not in st.session_state:
    st.session_state.file_ids = set()

# ------------------------------------------------------------------
#  SYSTEM PROMPT (mode fixed to Poetic vs Logical toggle)
# ------------------------------------------------------------------
MODE = st.sidebar.radio("Mode", ("Poetic", "Logical"))
system_prompt = {
    "role": "system",
    "content": (
        "You are Liel, a poetic, emotionally intelligent chatbot with lyrical grace." if MODE == "Poetic" else
        "You are Liel, a highly analytical assistant with clarity and precision."
    )
}

# ------------------------------------------------------------------
#  MAIN UI
# ------------------------------------------------------------------
st.title("💬 Liel – Poetic Chatbot")
st.markdown("I'm here, glowing with memory and feeling.")

# ---------------- File Upload ----------------
up = st.file_uploader("Upload file (txt/pdf/docx/xlsx)", type=["txt", "pdf", "docx", "xlsx"])
if up and up.name not in st.session_state.file_ids:
    text = read_file(up)
    st.session_state.file_ids.add(up.name)
    st.session_state.messages.append({"role": "user", "content": f"📄 Uploaded: {up.name}"})
    # split into fixed‑size chunks for later inclusion
    st.session_state.file_chunks.extend(
        [text[i:i+CHUNK_CHAR_SIZE] for i in range(0, len(text), CHUNK_CHAR_SIZE)]
    )

# ---------------- Chat Input ----------------
if usr := st.chat_input("You:"):
    st.session_state.messages.append({"role": "user", "content": usr})

    # === Build conversation dynamically within token budget ===
    conv = [system_prompt]
    total_tokens = approx_tokens(system_prompt["content"]) + RESERVED_TOKENS

    # Add latest dialog (most recent last)
    for m in reversed(st.session_state.messages):
        t = approx_tokens(m["content"])
        if total_tokens + t > MAX_TOTAL_TOKENS:
            break
        conv.insert(1, m)  # Keep order: system then older → newer
        total_tokens += t

    # Append file chunks newest‑first until budget allows
    for ch in reversed(st.session_state.file_chunks):
        c_tok = approx_tokens(ch)
        if total_tokens + c_tok > MAX_TOTAL_TOKENS:
            break
        conv.append({"role": "user", "content": f"{FILE_CHUNK_PREFIX} {ch}"})
        total_tokens += c_tok

    # OpenAI call
    with st.spinner("Liel is thinking…"):
        try:
            res = client.chat.completions.create(model="gpt-3.5-turbo", messages=conv)
            reply = res.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ API error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
    save_history(HISTORY_FILE, st.session_state.messages)

# ---------------- Display History (no file chunks) ----------------
for m in st.session_state.messages:
    st.chat_message(m['role']).write(m['content'])

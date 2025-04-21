import json
import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import pandas as pd

# === Constants ===
DEFAULT_HISTORY_KEEP = 50        # messages retained before summarization
DEFAULT_TEXT_CHUNK_SIZE = 5000   # characters per chunk sent to model
FILE_CHUNK_PREFIX = "[File chunk]"
SUMMARY_PREFIX = "**Summary:**"
HISTORY_FILE = "chat_history.json"

# === Streamlit config ===
st.set_page_config(page_title="Liel - Poetic Chatbot", layout="wide")

# === Sidebar controls ===
st.sidebar.title("⚙️ Settings")
HISTORY_KEEP = st.sidebar.slider("Max messages to keep", 10, 200, DEFAULT_HISTORY_KEEP, 10)
TEXT_CHUNK_SIZE = st.sidebar.slider("Max chars per file chunk", 1000, 20000, DEFAULT_TEXT_CHUNK_SIZE, 500)
MODE = st.sidebar.radio("Mode", ("Poetic", "Logical"))

# === OpenAI client ===
try:
    key = st.secrets.get("general", {}).get("OPENAI_API_KEY", "")
    if not key.startswith("sk-"):
        raise ValueError("OpenAI API key missing or invalid.")
    client = OpenAI(api_key=key)
except Exception as e:
    st.error(f"OpenAI init failed: {e}")
    st.stop()

# === Helper functions ===

def is_file_chunk(msg: dict) -> bool:
    """Return True if message represents a hidden file chunk (not for UI/history)."""
    content = msg.get("content", "")
    return content.startswith(FILE_CHUNK_PREFIX) or (msg.get("role") == "user" and len(content) > TEXT_CHUNK_SIZE)

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
        st.error(f"Save error: {e}")

@st.cache_data
def read_file(uploaded) -> str:
    try:
        if uploaded.type == "text/plain":
            return uploaded.getvalue().decode("utf-8")
        if uploaded.type == "application/pdf":
            return "\n".join(PdfReader(uploaded)[i].extract_text() or "" for i in range(len(PdfReader(uploaded))))
        if "wordprocessingml.document" in uploaded.type:
            return "\n".join(p.text for p in docx.Document(uploaded).paragraphs)
        if "spreadsheetml.sheet" in uploaded.type:
            return pd.read_excel(uploaded).to_csv(index=False, sep="\t")
    except Exception as e:
        st.error(f"File read error: {e}")
    return ""


def summarize(msgs: list) -> list:
    if len(msgs) <= HISTORY_KEEP:
        return msgs
    old, recent = msgs[:-HISTORY_KEEP], msgs[-HISTORY_KEEP:]
    prompt = "Summarize:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in old)
    try:
        res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role":"system","content":prompt}])
        summ = res.choices[0].message.content
        return [{"role":"assistant","content":f"{SUMMARY_PREFIX} {summ}"}] + recent
    except Exception:
        return msgs

# === Session state ===
if "messages" not in st.session_state:
    st.session_state.messages = summarize(load_history(HISTORY_FILE))
else:
    st.session_state.messages = [m for m in st.session_state.messages if not is_file_chunk(m)]

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "file_ids" not in st.session_state:
    st.session_state.file_ids = set()

# === System prompt ===
system_prompt = {
    "role":"system",
    "content":(
        "You are Liel, a poetic, emotionally intelligent chatbot with lyrical grace." if MODE=="Poetic"
        else "You are Liel, a highly analytical assistant with clarity and precision."
    )
}

# === UI ===
st.title("💬 Liel - Poetic Chatbot")
st.markdown("I'm here, glowing with memory and feeling.")

# File uploader
up = st.file_uploader("Upload file (txt, pdf, docx, xlsx)", type=["txt","pdf","docx","xlsx"])
if up and up.name not in st.session_state.file_ids:
    text = read_file(up)
    for i in range(0, len(text), TEXT_CHUNK_SIZE):
        st.session_state.chunks.append(text[i:i+TEXT_CHUNK_SIZE])
    st.session_state.file_ids.add(up.name)
    st.session_state.messages.append({"role":"user","content":f"📄 Uploaded: {up.name}"})

# Chat
if prompt := st.chat_input("You:"):
    st.session_state.messages.append({"role":"user","content":prompt})
    conv = [system_prompt] + st.session_state.messages + [{"role":"user","content":f"{FILE_CHUNK_PREFIX} {c}"} for c in st.session_state.chunks]
    st.session_state.chunks.clear()
    with st.spinner("Liel is thinking..."):
        try:
            res = client.chat.completions.create(model="gpt-3.5-turbo", messages=conv)
            resp_text = res.choices[0].message.content
        except Exception as e:
            resp_text = f"⚠️ API error: {e}"
    st.session_state.messages.append({"role":"assistant","content":resp_text})
    st.chat_message("assistant").write(resp_text)
    st.session_state.messages = summarize(st.session_state.messages)
    save_history(HISTORY_FILE, st.session_state.messages)

# Display
for m in st.session_state.messages:
    if is_file_chunk(m):
        continue
    st.chat_message(m['role']).write(m['content'])

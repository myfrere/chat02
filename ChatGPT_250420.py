import json
import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import pandas as pd

# === Constants & Defaults ===
DEFAULT_HISTORY_KEEP = 50
DEFAULT_TEXT_CHUNK_SIZE = 5000
FILE_CHUNK_PREFIX = "[File chunk]"
SUMMARY_PREFIX = "**Summary:**"
HISTORY_FILE = "chat_history.json"

# === Streamlit page config ===
st.set_page_config(page_title="Liel - Poetic Chatbot", layout="wide")

# === Sidebar Controls ===
st.sidebar.title("⚙️ Settings")
history_keep = st.sidebar.slider(
    "Max messages to keep before summarizing", 10, 200, DEFAULT_HISTORY_KEEP, 10
)
text_chunk_size = st.sidebar.slider(
    "Max characters per file chunk", 1000, 20000, DEFAULT_TEXT_CHUNK_SIZE, 500
)
mode = st.sidebar.radio("Mode", ("Poetic", "Logical"))

# === OpenAI Client Initialization ===
try:
    api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY", "")
    if not api_key.startswith("sk-"):
        raise ValueError("Invalid or missing OpenAI API key.")
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"OpenAI init failed: {e}")
    st.stop()

# === Helper Functions ===
def is_file_chunk_message(msg: dict) -> bool:
    """Return True if message is a file chunk marker."""
    content = msg.get("content", "")
    return content.startswith(FILE_CHUNK_PREFIX)

@st.cache_data
def load_history(path: str) -> list:
    """Load saved history, filtering out chunk markers."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                msgs = json.load(f)
            # filter out chunk markers
            return [m for m in msgs if not is_file_chunk_message(m)]
        except Exception:
            return []
    return []


def save_history(path: str, msgs: list):
    """Save history excluding chunk markers."""
    try:
        filtered = [m for m in msgs if not is_file_chunk_message(m)]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving history: {e}")

@st.cache_data
def read_uploaded_file(uploaded) -> str:
    """Extract plain text from uploaded file."""
    try:
        if uploaded.type == "text/plain":
            return uploaded.getvalue().decode("utf-8")
        if uploaded.type == "application/pdf":
            reader = PdfReader(uploaded)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        if "wordprocessingml.document" in uploaded.type:
            doc = docx.Document(uploaded)
            return "\n".join(p.text for p in doc.paragraphs)
        if "spreadsheetml.sheet" in uploaded.type:
            df = pd.read_excel(uploaded)
            return df.to_csv(index=False, sep="\t")
    except Exception as e:
        st.error(f"File read error: {e}")
    return ""


def summarize_history(msgs: list) -> list:
    """Summarize old messages beyond history_keep."""
    if len(msgs) <= history_keep:
        return msgs
    old, recent = msgs[:-history_keep], msgs[-history_keep:]
    prompt = (
        "Summarize the following conversation, keeping key points:\n"
        + "\n".join(f"{m['role']}: {m['content']}" for m in old)
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        summary = resp.choices[0].message.content
        return [{"role": "assistant", "content": f"{SUMMARY_PREFIX} {summary}"}] + recent
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return msgs

# === Initialize session state ===
if "messages" not in st.session_state:
    st.session_state.messages = summarize_history(load_history(HISTORY_FILE))
if "file_chunks" not in st.session_state:
    st.session_state.file_chunks = []
if "file_ids" not in st.session_state:
    st.session_state.file_ids = set()

# === System prompt ===
system_message = {
    "role": "system",
    "content": (
        "You are Liel, a poetic, emotionally intelligent chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
        if mode == "Poetic"
        else
        "You are Liel, a highly analytical and logical assistant who solves complex tasks with clarity and precision."
    )
}

# === Main UI ===
st.title("💬 Liel - Poetic Chatbot")
st.markdown("I'm here, glowing with memory and feeling.")

# --- File Upload Handling ---
uploaded_file = st.file_uploader(
    "Upload file (txt, pdf, docx, xlsx)", type=["txt","pdf","docx","xlsx"]
)
if uploaded_file and uploaded_file.name not in st.session_state.file_ids:
    text = read_uploaded_file(uploaded_file)
    # chunk for API input only
    for i in range(0, len(text), text_chunk_size):
        st.session_state.file_chunks.append(text[i:i+text_chunk_size])
    st.session_state.file_ids.add(uploaded_file.name)
    # record an icon-only message
    st.session_state.messages.append({"role": "user", "content": f"📄 Uploaded: {uploaded_file.name}"})

# --- Chat Input and Response ---
if user_input := st.chat_input("You:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    # build conversation including file chunks
    conv = [system_message] + st.session_state.messages
    for chunk in st.session_state.file_chunks:
        conv.append({"role": "user", "content": f"{FILE_CHUNK_PREFIX} {chunk}"})
    # clear after use
    st.session_state.file_chunks.clear()
    # call OpenAI
    with st.spinner("Liel is thinking..."):
        try:
            resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=conv)
            reply = resp.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ API call failed: {e}"
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
    # summarize & save history
    st.session_state.messages = summarize_history(st.session_state.messages)
    save_history(HISTORY_FILE, st.session_state.messages)

# --- Display Chat History ---
for msg in st.session_state.messages:
    # only display actual messages, not raw chunks
    if is_file_chunk_message(msg):
        continue
    st.chat_message(msg['role']).write(msg['content'])

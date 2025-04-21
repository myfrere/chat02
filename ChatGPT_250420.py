import json
import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import pandas as pd

# 🍃 Streamlit page configuration
st.set_page_config(page_title="Liel - Poetic Chatbot", layout="wide")

# 🔐 OpenAI client initialization
try:
    api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY", "")
    if not api_key or not api_key.startswith("sk-"):
        raise ValueError("Invalid or missing OpenAI API Key.")
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize OpenAI: {e}")
    st.stop()

# 📦 History management
HISTORY_FILE = "chat_history.json"
MAX_HISTORY_KEEP = 50

@st.cache_data
def load_history(path: str) -> list:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_history(path: str, msgs: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(msgs, f, ensure_ascii=False, indent=2)

# 📄 File reading with chunking support
MAX_TEXT_LENGTH = 5000

def read_uploaded_file(uploaded) -> str:
    try:
        if uploaded.type == "text/plain":
            return uploaded.getvalue().decode('utf-8')
        if uploaded.type == "application/pdf":
            reader = PdfReader(uploaded)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        if "wordprocessingml.document" in uploaded.type:
            doc = docx.Document(uploaded)
            return "\n".join(p.text for p in doc.paragraphs)
        if "spreadsheetml.sheet" in uploaded.type:
            df = pd.read_excel(uploaded)
            return df.to_csv(index=False, sep='\t')
    except Exception as e:
        st.error(f"File read error: {e}")
    return ""

# 📝 Summarize old history

def summarize_history(msgs: list) -> list:
    if len(msgs) <= MAX_HISTORY_KEEP:
        return msgs
    old, recent = msgs[:-MAX_HISTORY_KEEP], msgs[-MAX_HISTORY_KEEP:]
    prompt = (
        "Summarize the following conversation, keeping key points and discarding trivial talk:\n" +
        "\n".join(f"{m['role']}: {m['content']}" for m in old)
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        summary = resp.choices[0].message.content
        return [{"role": "assistant", "content": f"**Summary:** {summary}"}] + recent
    except Exception:
        return msgs

# ↪️ Init session state
if "messages" not in st.session_state:
    st.session_state.messages = load_history(HISTORY_FILE)
    st.session_state.messages = summarize_history(st.session_state.messages)
if "file_ids" not in st.session_state:
    st.session_state.file_ids = []

# 🔄 System prompt based on mode
mode = st.sidebar.radio("Mode", ("Poetic", "Logical"))
system_message = {
    "role": "system",
    "content": (
        "You are Liel, a poetic, emotionally intelligent chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
        if mode == "Poetic"
        else
        "You are Liel, a highly analytical and logical assistant who solves complex tasks with clarity and precision."
    )
}

# 🌐 App title and intro
st.title("💬 Liel - Poetic Chatbot")
st.markdown("I'm here, glowing with memory and feeling.")

# 📁 File uploader (process once per file)
uploaded_file = st.file_uploader("Upload file (txt, pdf, docx, xlsx)", type=["txt","pdf","docx","xlsx"])
if uploaded_file and uploaded_file.name not in st.session_state.file_ids:
    raw_text = read_uploaded_file(uploaded_file)
    for start in range(0, len(raw_text), MAX_TEXT_LENGTH):
        part = raw_text[start:start+MAX_TEXT_LENGTH]
        st.session_state.messages.append({"role": "user", "content": f"[File chunk]\n{part}"})
    st.session_state.file_ids.append(uploaded_file.name)

# 🗨️ Display conversation history
for msg in st.session_state.messages:
    role = "user" if msg['role'] == 'user' else "assistant"
    st.chat_message(role).write(msg['content'])

# 💬 Chat input and response
if user_input := st.chat_input("You:"):
    # append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # prepare conversation
    conversation = [system_message] + st.session_state.messages

    # get AI response
    with st.spinner("Liel is thinking..."):
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=conversation)
    answer = response.choices[0].message.content

    # append and display answer
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

    # summarize & save history
    st.session_state.messages = summarize_history(st.session_state.messages)
    save_history(HISTORY_FILE, st.session_state.messages)

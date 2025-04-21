import json
import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import pandas as pd

# 🍃 Streamlit page configuration
st.set_page_config(page_title="Liel - Poetic Chatbot", layout="wide")

# 🔐 OpenAI client initialization with secured secrets handling
try:
    api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY", "")
    if not api_key or not api_key.startswith("sk-"):
        raise ValueError("Invalid or missing OpenAI API Key. Please check your configuration.")
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to initialize OpenAI: {e}")
    st.stop()

# 📦 Load/Save conversation history
HISTORY_FILE = "chat_history.json"
MAX_HISTORY_KEEP = 50  # keep this many recent messages after summarizing

@st.cache_data
def load_history(path: str) -> list:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"JSON decoding error: {e}")
    return []

def save_history(path: str, msgs: list):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(msgs, f, ensure_ascii=False, indent=2)
    except IOError as e:
        st.error(f"Error saving history: {e}")

# 📄 Read uploaded files (with chunking)
MAX_TEXT_LENGTH = 5000  # max chars per file chunk

def read_uploaded_file(uploaded) -> str:
    try:
        text = ""
        if uploaded.type == "text/plain":
            text = uploaded.getvalue().decode('utf-8')
        elif uploaded.type == "application/pdf":
            reader = PdfReader(uploaded)
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        elif "wordprocessingml.document" in uploaded.type:
            doc = docx.Document(uploaded)
            text = "\n".join(p.text for p in doc.paragraphs)
        elif "spreadsheetml.sheet" in uploaded.type:
            df = pd.read_excel(uploaded)
            text = df.to_csv(index=False, sep='\t')

        return text
    except Exception as e:
        st.error(f"파일 읽기 중 오류: {e}")
        return ""

# 📝 Summarize old history to keep file size manageable

def summarize_history(msgs: list, client: OpenAI, keep: int = MAX_HISTORY_KEEP) -> list:
    if len(msgs) <= keep:
        return msgs
    old = msgs[:-keep]
    recent = msgs[-keep:]
    prompt = (
        "Summarize the following conversation, preserving key information and discarding trivial talk:\n"
        + "\n".join(f"{m['role']}: {m['content']}" for m in old)
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":prompt}]
        )
        summary = resp.choices[0].message.content
        return [{"role":"assistant","content":f"**Summary of earlier conversation:** {summary}"}] + recent
    except Exception as e:
        st.error(f"Error summarizing history: {e}")
        return msgs

# ↪️ Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = load_history(HISTORY_FILE)
    st.session_state.messages = summarize_history(st.session_state.messages, client)

# 🔄 Mode selection & system message definition
mode = st.sidebar.radio("모드 선택", ("Poetic", "Logical"))
system_message = {
    "role": "system",
    "content": (
        "You are Liel, a poetic, emotionally intelligent chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
        if mode == "Poetic"
        else
        "You are Liel, a highly analytical and logical assistant who solves complex tasks with clarity and precision."
    )
}

# 🌐 Page UI
st.title("💬 Liel - Poetic Chatbot")
st.markdown("I'm here, glowing with memory and feeling.")

# Display conversation history
for m in st.session_state.messages:
    label = "You:" if m['role'] == 'user' else "Liel:"
    st.text_area(label, value=m['content'], height=120, disabled=True)

# Chat form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("You:", height=120)
    uploaded_file = st.file_uploader("📁 파일 업로드 (txt, pdf, docx, xlsx)", type=["txt","pdf","docx","xlsx"])
    submitted = st.form_submit_button("전송")

if submitted:
    # Process file in chunks
    if uploaded_file:
        raw = read_uploaded_file(uploaded_file)
        chunks = [raw[i:i+MAX_TEXT_LENGTH] for i in range(0, len(raw), MAX_TEXT_LENGTH)]
        for idx, ch in enumerate(chunks, 1):
            st.session_state.messages.append({
                "role": "user",
                "content": f"[파일 조각 {idx}/{len(chunks)}]\n{ch}"
            })
    # Append user text
    if user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    # Compose and send
    msgs = [system_message] + st.session_state.messages
    try:
        with st.spinner("💬 Liel이 응답 중..."):
            resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=msgs)
        reply = resp.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"⚠️ OpenAI API 호출 중 오류 발생: {e}")

    # Summarize if too long and save
    st.session_state.messages = summarize_history(st.session_state.messages, client)
    save_history(HISTORY_FILE, st.session_state.messages)

    # Rerun to display updated history
    st.experimental_rerun()

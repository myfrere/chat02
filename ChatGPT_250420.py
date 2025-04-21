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
        raise ValueError("Invalid or missing OpenAI API Key. Please check your configuration.")
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"❌ Failed to initialize OpenAI: {e}")
    st.stop()

# 📦 Load/Save conversation history
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

# 📄 File reader
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
        st.error(f"파일 읽기 오류: {e}")
    return ""

# 📝 Summarize history

def summarize_history(msgs: list) -> list:
    if len(msgs) <= MAX_HISTORY_KEEP:
        return msgs
    old, recent = msgs[:-MAX_HISTORY_KEEP], msgs[-MAX_HISTORY_KEEP:]
    prompt = (
        "Summarize conversation preserving key points and discarding trivial talk:\n" +
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

# ↪️ Init session
if "messages" not in st.session_state:
    st.session_state.messages = load_history(HISTORY_FILE)
    st.session_state.messages = summarize_history(st.session_state.messages)

# 🔄 Mode & system message
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

# 🌐 UI
st.title("💬 Liel - Poetic Chatbot")
st.markdown("I'm here, glowing with memory and feeling.")

# Display chat history
for msg in st.session_state.messages:
    role_str = "user" if msg['role'] == 'user' else "assistant"
    st.chat_message(role_str).write(msg['content'])

# Chat input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("You:", height=120, key="user_input")
    uploaded_file = st.file_uploader("Upload file (txt, pdf, docx, xlsx)", type=["txt","pdf","docx","xlsx"])
    submitted = st.form_submit_button("Send")

if submitted:
    # Process file
    if uploaded_file:
        raw_text = read_uploaded_file(uploaded_file)
        chunks = [raw_text[i:i+MAX_TEXT_LENGTH] for i in range(0, len(raw_text), MAX_TEXT_LENGTH)]
        for idx, chunk in enumerate(chunks, 1):
            st.session_state.messages.append({
                "role": "user",
                "content": f"[File part {idx}/{len(chunks)}]\n{chunk}"
            })
    # Process user input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

    # Call OpenAI
    conversation = [system_message] + st.session_state.messages
    try:
        with st.spinner("Liel is thinking..."):
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=conversation)
        assistant_reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        # Immediately display the assistant's reply
        st.chat_message("assistant").write(assistant_reply)
    except Exception as e:
        st.error(f"API error: {e}")

    # Summarize & save history
    st.session_state.messages = summarize_history(st.session_state.messages)
    save_history(HISTORY_FILE, st.session_state.messages)

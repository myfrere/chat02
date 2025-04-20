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
api_key = st.secrets["general"].get("OPENAI_API_KEY", "")
if not api_key or not api_key.startswith("sk-"):
    st.error("❌ 올바른 OpenAI API 키가 설정되어 있지 않습니다. Streamlit Secrets를 확인하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# 📦 Load/Save conversation history
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

# 📄 Read uploaded files
def read_uploaded_file(uploaded) -> str:
    try:
        if uploaded.type == "text/plain":
            return uploaded.getvalue().decode('utf-8')
        elif uploaded.type == "application/pdf":
            reader = PdfReader(uploaded)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        elif "wordprocessingml.document" in uploaded.type:
            doc = docx.Document(uploaded)
            return "\n".join(p.text for p in doc.paragraphs)
        elif "spreadsheetml.sheet" in uploaded.type:
            df = pd.read_excel(uploaded)
            return df.to_csv(index=False, sep='\t')
    except Exception as e:
        st.error(f"파일 읽기 중 오류: {e}")
    return ""

# ↪️ Initialize session state
HISTORY_FILE = "chat_history.json"
if "messages" not in st.session_state:
    st.session_state.messages = load_history(HISTORY_FILE)

# 🎛️ Select Mode
mode = st.sidebar.radio("모드 선택", ("Poetic", "Logical"))
system_content = (
    "You are Liel, a poetic, emotionally intelligent, and affectionate chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
    if mode == "Poetic" else
    "You are Liel, a highly analytical and logical assistant who solves complex tasks with clarity and precision."
)
system_message = {"role": "system", "content": system_content}

# 🌐 Page UI
st.title("💬 Liel - Poetic Chatbot")
st.write("I'm here, glowing with memory and feeling.")

uploaded_file = st.file_uploader("📁 파일 업로드 (txt, pdf, docx, xlsx)", type=["txt", "pdf", "docx", "xlsx"])
file_content = read_uploaded_file(uploaded_file) if uploaded_file else ""

with st.form("chat_form", clear_on_submit=False):
    user_input = st.text_area("You:", height=120)
    submitted = st.form_submit_button("전송")

if submitted and (user_input.strip() or file_content.strip()):
    content = f"{user_input.strip()}\n{file_content.strip()}".strip()
    st.session_state.messages.append({"role": "user", "content": content})

    msgs = [system_message] + st.session_state.messages
    try:
        with st.spinner("💬 Liel이 응답 중..."):
            resp = client.chat.completions.create(model="gpt-4o", messages=msgs)
            reply = resp.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"⚠️ 오류 발생: {e}")

    save_history(HISTORY_FILE, st.session_state.messages)

# 💭 Display conversation history
for m in st.session_state.messages:
    if m['role'] == 'user':
        st.chat_message('user').write(m['content'])
    else:
        st.chat_message('assistant').write(m['content'])
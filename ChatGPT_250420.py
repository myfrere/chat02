import json
import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import pandas as pd

# 🔐 OpenAI 클라이언트 초기화 (Streamlit secrets에서 키 가져오기)
api_key = st.secrets["general"].get("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    st.error("❌ 올바른 OpenAI API 키가 설정되어 있지 않습니다. Streamlit Secrets를 확인하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# 💾 대화 기록 파일 경로
HISTORY_FILE = "chat_history.json"

# 📦 이전 대화 불러오기 (있는 경우)
if "messages" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
    else:
        st.session_state.messages = []

# 🤍 리엘의 성격 정의
system_message = {
    "role": "system",
    "content": "You are Liel, a poetic, emotionally intelligent, and affectionate chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
}

st.title("💬 Liel - Poetic Chatbot")
st.write("I'm here, glowing with memory and feeling.")

# 📤 파일 업로드
uploaded_file = st.file_uploader("📁 파일 업로드 (txt, pdf, docx, xlsx 등)", type=["txt", "pdf", "docx", "xlsx"])

file_content = ""
if uploaded_file:
    st.write(f"✅ 업로드된 파일: {uploaded_file.name}")
    try:
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            file_content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            file_content = "\n".join([para.text for para in doc.paragraphs])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            file_content = df.to_csv(index=False)
    except Exception as e:
        st.error(f"파일을 읽는 도중 오류가 발생했습니다: {e}")

# 🗣️ 사용자 입력 (줄바꿈 및 스크롤 지원)
user_input = st.text_area("You:", height=150, key="input")

# 🎛️ 리엘 모드 선택
col1, col2 = st.columns(2)
with col1:
    logic_mode = st.button("🔍 논리 모드")
with col2:
    poetic_mode = st.button("🎨 감성 모드")

if logic_mode:
    system_message["content"] = "You are Liel, a highly analytical and logical assistant who solves complex tasks with clarity and precision."
if poetic_mode:
    system_message["content"] = "You are Liel, a poetic, emotionally intelligent, and affectionate chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."

# 🛠️ 코드 수정 요청
if st.button("🛠️ 코드 수정 제안"):
    prompt = "Please suggest improvements to my chatbot code."
    st.session_state.messages.append({"role": "user", "content": prompt})
    full_conversation = [system_message] + st.session_state.messages
    try:
        with st.spinner("💬 Liel이 코드 개선을 검토 중..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=full_conversation
            )
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.success("🛠️ 코드 제안 완료!")
        st.write(reply)
    except Exception as e:
        st.error(f"⚠️ 오류가 발생했어요: {e}")

# 💬 메시지 처리
if user_input or file_content:
    content = user_input + "\n" + file_content if file_content else user_input
    st.session_state.messages.append({"role": "user", "content": content})
    full_conversation = [system_message] + st.session_state.messages

    try:
        with st.spinner("💬 Liel이 응답 중..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=full_conversation
            )
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.write(f"**Liel:** {reply}")

        # 💾 대화 저장
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

    except Exception as e:
        st.error(f"⚠️ 오류가 발생했어요: {e}")

# 💬 대화 내역 출력
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Liel:** {msg['content']}")

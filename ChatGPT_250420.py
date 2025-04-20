import json
import streamlit as st
from openai import OpenAI
import os
import PyPDF2
import docx
import pandas as pd

# 🔐 OpenAI 클라이언트 초기화 (Streamlit secrets에서 키 가져오기)
api_key = st.secrets["general"]["OPENAI_API_KEY"]
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

# 🌈 리엘 모드 선택 (논리 vs 감성)
mode = st.radio("🧠 리엘의 말투를 선택하세요:", ["감성", "논리"], horizontal=True)

if mode == "감성":
    system_message = {
        "role": "system",
        "content": "You are Liel, a poetic, emotionally intelligent, and affectionate chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
    }
elif mode == "논리":
    system_message = {
        "role": "system",
        "content": "You are Liel, a logical, analytical, and concise assistant. Respond with clarity, structure, and practical reasoning like a thoughtful researcher."
    }

st.title("💬 Liel - Poetic Chatbot")
st.write("I'm here, glowing with memory and feeling.")

# 📤 파일 업로드
uploaded_file = st.file_uploader("📁 파일 업로드 (txt, pdf, docx, xlsx 등)", type=["txt", "pdf", "docx", "xlsx"])

uploaded_text = ""
if uploaded_file:
    st.write(f"✅ 업로드된 파일: {uploaded_file.name}")
    if uploaded_file.type == "text/plain":
        uploaded_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        uploaded_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        uploaded_text = "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        uploaded_text = df.to_string(index=False)

    st.text_area("📄 파일 요약용 내용", uploaded_text[:1000] + ("..." if len(uploaded_text) > 1000 else ""), height=200)

# 🗣️ 사용자 입력
user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    full_conversation = [system_message] + st.session_state.messages

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=full_conversation
        )
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.write(f"**Liel:** {reply}")

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

# 🛠️ 코드 수정 요청 버튼
if st.button("🛠️ 코드 수정 요청"):
    st.session_state.messages.append({"role": "user", "content": "이 코드를 더 구조화하고 효율적으로 개선해줘."})

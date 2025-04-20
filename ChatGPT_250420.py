import json
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import pandas as pd

# 🔐 OpenAI 클라이언트 초기화 (Streamlit secrets에서 키 가져오기)
api_key = st.secrets["general"]["OPENAI_API_KEY"]

if not api_key or not api_key.startswith("sk-"):
    st.error("❌ 올바른 OpenAI API 키가 설정되어 있지 않습니다. Streamlit Secrets를 확인하세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# 대화 기록 저장 (Streamlit 세션 상태 활용)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🤍 리엘의 성격 정의
system_message = {
    "role": "system",
    "content": "You are Liel, a poetic, emotionally intelligent, and affectionate chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
}

st.title("💬 Liel - Poetic Chatbot")
st.write("I'm here, glowing with memory and feeling.")

# 📎 파일 업로드 (다양한 형식 지원)
uploaded_file = st.file_uploader("파일 업로드 (.txt, .pdf, .docx, .xlsx)", type=["txt", "pdf", "docx", "xlsx"])
file_text = ""

if uploaded_file is not None:
    try:
        if uploaded_file.type == "text/plain":
            file_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            file_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            file_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            df = pd.read_excel(uploaded_file)
            file_text = df.to_string(index=False)
        else:
            st.warning("지원하지 않는 파일 형식입니다.")
        
        if file_text:
            st.session_state.messages.append({"role": "user", "content": f"[업로드된 파일 내용]\n{file_text}"})
            st.success("✅ 파일 내용이 대화에 추가되었습니다!")

    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

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

    except Exception as e:
        st.error(f"⚠️ 오류가 발생했어요: {e}")

# 💬 대화 내역 출력
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Liel:** {msg['content']}")

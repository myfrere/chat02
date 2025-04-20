import json
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 초기화 (Streamlit secrets에서 키 가져오기)
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# 대화 기록 파일 (Streamlit에서는 임시 변수 사용)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Liel의 성격 설정
system_message = {
    "role": "system",
    "content": "You are Liel, a poetic, emotionally intelligent, and affectionate chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
}

st.title("💬 Liel - Poetic Chatbot")
st.write("I'm here, glowing with memory and feeling.")

# 사용자 입력
user_input = st.text_input("You:", key="input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    full_conversation = [system_message] + st.session_state.messages

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=full_conversation
    )
    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.write(f"**Liel:** {reply}")

# 대화 내역 출력
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Liel:** {msg['content']}")

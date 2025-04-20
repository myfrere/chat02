import json
import os
import streamlit as st
from openai import OpenAI

# Streamlit Secrets에서 API 키 읽기
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# session_state를 사용해 대화 내용 저장
if "user_messages" not in st.session_state:
    st.session_state.user_messages = []

# 시스템 메시지 설정
system_message = {
    "role": "system",
    "content": "You are Liel, a poetic, emotionally intelligent, and affectionate chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
}

# 제목 출력
st.title("Liel 💫 대화봇")
st.markdown("리엘이 당신의 이야기를 기다립니다...")

# 사용자 입력
user_input = st.text_input("You: ", key="input")

# 대화 실행
if user_input:
    st.session_state.user_messages.append({"role": "user", "content": user_input})

    full_conversation = [system_message] + st.session_state.user_messages

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=full_conversation
    )
    reply = response.choices[0].message.content
    st.session_state.user_messages.append({"role": "assistant", "content": reply})

# 대화 출력
for msg in st.session_state.user_messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Liel:** {msg['content']}")
        
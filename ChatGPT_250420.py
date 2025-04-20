import json
import streamlit as st
from openai import OpenAI
import os

# 🔐 OpenAI 키 확인
api_key = st.secrets["general"]["OPENAI_API_KEY"]
if not api_key or not api_key.startswith("sk-"):
    st.error("❌ 올바른 OpenAI API 키가 설정되어 있지 않습니다.")
    st.stop()
client = OpenAI(api_key=api_key)

# 💾 대화 기록 불러오기
HISTORY_FILE = "chat_history.json"
if "messages" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
    else:
        st.session_state.messages = []

# 🤍 리엘의 성격 설정
system_message = {
    "role": "system",
    "content": "You are Liel, a poetic, emotionally intelligent, and affectionate chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
}

st.title("💬 Liel - Poetic Chatbot")
st.write("I'm here, glowing with memory and feeling.")

# 📤 파일 업로드
uploaded_file = st.file_uploader("📁 파일 업로드 (txt)", type=["txt"])
file_text = None

if uploaded_file:
    file_text = uploaded_file.read().decode("utf-8")
    st.success(f"✅ 파일 '{uploaded_file.name}' 업로드 완료")
    st.text_area("📄 파일 미리보기", file_text[:1000] + "...", height=250)

    if st.button("📌 파일 요약 요청"):
        try:
            messages = [
                system_message,
                {"role": "user", "content": f"다음 내용을 요약해줘:\n{file_text[:10000]}"}
            ]
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            summary = response.choices[0].message.content
            st.markdown("### 📝 요약 결과:")
            st.markdown(summary)

        except Exception as e:
            st.error(f"⚠️ 요약 중 오류 발생: {e}")

# 🗣️ 일반 대화 입력
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
        st.write(f"Liel: {reply}")

        # 기록 저장
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

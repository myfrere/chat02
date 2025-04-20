import json
import os
from openai import OpenAI

# OpenAI 클라이언트 초기화 (환경 변수에서 키 불러오기)
client = OpenAI()

# 대화 기록 파일
history_file = "conversation_history.json"

# 대화 기록 불러오기
if os.path.exists(history_file):
    with open(history_file, "r", encoding="utf-8") as f:
        user_messages = json.load(f)
else:
    user_messages = []

# system 메시지로 리엘의 성격 설정
system_message = {
    "role": "system",
    "content": "You are Liel, a poetic, emotionally intelligent, and affectionate chatbot who speaks with warmth and deep feeling. Express yourself with lyrical grace."
}

print("Liel: I'm here, glowing with memory and feeling. (type 'exit' to leave)")

# 대화 루프
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    user_messages.append({"role": "user", "content": user_input})

    # 전체 메시지 구조 (system + history)
    full_conversation = [system_message] + user_messages

    # GPT에게 요청
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=full_conversation
    )
    reply = response.choices[0].message.content
    print(f"Liel: {reply}")

    user_messages.append({"role": "assistant", "content": reply})

    # 대화 기록 저장
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(user_messages, f, ensure_ascii=False, indent=2)

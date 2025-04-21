import json
import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import pandas as pd
from time import sleep

# ------------------------------------------------------------------
# CONFIG & CONSTANTS
# ------------------------------------------------------------------
MAX_TOTAL_TOKENS = 12_000      # safe cap (gpt‑3.5‑turbo context ≤ 16 385)
CHUNK_CHAR_SIZE  = 2_000       # characters per chunk for summarization
RESERVED_TOKENS  = 1_000       # buffer for reply/system
SUMMARY_PREFIX   = "**Summary:**"
HISTORY_FILE     = "chat_history.json"

# ------------------------------------------------------------------
# PAGE & OPENAI CLIENT
# ------------------------------------------------------------------
st.set_page_config(page_title="Liel – Poetic Chatbot", layout="wide")

try:
    api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY", "")
    if not api_key.startswith("sk-"):
        raise ValueError("Invalid OpenAI API key.")
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"❌ OpenAI init failed: {e}")
    st.stop()

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def approx_tokens(txt: str) -> int:
    return max(1, len(txt) // 3)  # 3 chars ≈ 1 token upper‑bound

@st.cache_data
def read_file(file) -> str:
    try:
        if file.type == "text/plain":
            return file.getvalue().decode("utf-8")
        if file.type == "application/pdf":
            return "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
        if "wordprocessingml.document" in file.type:
            return "\n".join(p.text for p in docx.Document(file).paragraphs)
        if "spreadsheetml.sheet" in file.type:
            return pd.read_excel(file).to_csv(index=False, sep="\t")
    except Exception as e:
        st.error(f"File read error: {e}")
    return ""

@st.cache_data(show_spinner=False)
def load_history(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(path: str, msgs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(msgs, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_history(HISTORY_FILE)
if "doc_summaries" not in st.session_state:
    st.session_state.doc_summaries = {}   # {filename: summary_str}
if "file_ids" not in st.session_state:
    st.session_state.file_ids = set()

# ------------------------------------------------------------------
# MODE & SYSTEM PROMPT
# ------------------------------------------------------------------
MODE = st.sidebar.radio("Mode", ("Poetic", "Logical"))
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Liel, a poetic, emotionally intelligent chatbot with lyrical grace."
        if MODE == "Poetic"
        else "You are Liel, a highly analytical assistant with clarity and precision."
    ),
}

# ------------------------------------------------------------------
# UI HEADER
# ------------------------------------------------------------------
st.title("💬 Liel – Poetic Chatbot")
st.markdown("I'm here, glowing with memory and feeling.")

# ------------------------------------------------------------------
# UPLOAD & AUTO‑SUMMARIZE
# ------------------------------------------------------------------
CHUNK_PROMPT = "Summarize this chunk in 2‒3 short Korean bullet points."  # adjust for language

def summarize_chunks(chunks):
    summaries = []
    prog = st.progress(0, text="Summarizing uploaded file…")
    step = 1 / max(1, len(chunks))
    for i, ch in enumerate(chunks, 1):
        try:
            res = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": CHUNK_PROMPT},
                    {"role": "user", "content": ch},
                ],
            )
            summaries.append(res.choices[0].message.content.strip())
        except Exception as e:
            summaries.append("(요약 실패)" + str(e))
        prog.progress(min(1.0, i * step))
        sleep(0.1)
    prog.empty()
    return "\n".join(summaries)

uploaded = st.file_uploader("Upload file (txt/pdf/docx/xlsx)", type=["txt","pdf","docx","xlsx"])
if uploaded and uploaded.name not in st.session_state.file_ids:
    full_text = read_file(uploaded)
    chunks = [full_text[i:i+CHUNK_CHAR_SIZE] for i in range(0, len(full_text), CHUNK_CHAR_SIZE)]
    summary_text = summarize_chunks(chunks)
    st.session_state.doc_summaries[uploaded.name] = summary_text
    st.session_state.file_ids.add(uploaded.name)
    st.session_state.messages.append({"role": "user", "content": f"📄 Uploaded & summarized: {uploaded.name}"})

# ------------------------------------------------------------------
# CHAT INPUT
# ------------------------------------------------------------------
if user_prompt := st.chat_input("You:"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Build conversation under token budget
    conv = [SYSTEM_PROMPT]
    budget = approx_tokens(SYSTEM_PROMPT["content"]) + RESERVED_TOKENS

    # 1) include doc summaries (newest first)
    for name, summ in list(st.session_state.doc_summaries.items())[::-1]:
        tok = approx_tokens(summ)
        if budget + tok > MAX_TOTAL_TOKENS:
            continue
        conv.append({"role": "system", "content": f"[Doc {name} summary]
{summ}"})
        budget += tok

    # 2) include recent chat (newest last)
    for msg in reversed(st.session_state.messages):
        tok = approx_tokens(msg["content"])
        if budget + tok > MAX_TOTAL_TOKENS:
            break
        conv.insert(1, msg)
        budget += tok

    # 3) final hard‑trim if still over limit
    while sum(approx_tokens(m["content"]) for m in conv) > MAX_TOTAL_TOKENS and len(conv) > 1:
        # drop the second element (oldest non‑system message)
        conv.pop(1)

    with st.spinner("Thinking…"):​("Thinking…"):
        try:
            res = client.chat.completions.create(model="gpt-3.5-turbo", messages=conv)
            answer = res.choices[0].message.content
        except Exception as e:
            answer = f"⚠️ API error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
    save_history(HISTORY_FILE, st.session_state.messages)

# ------------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------------
for m in st.session_state.messages:
    st.chat_message(m['role']).write(m['content'])

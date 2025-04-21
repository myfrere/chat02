import json
import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfFileReader
import docx
import pandas as pd
from time import sleep

# ------------------------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------------------------
# Available models: gpt-3.5-turbo, gpt-4, gpt-4-32k, gpt-4o
MODEL = st.sidebar.selectbox(
    'Select model',
    ('gpt-3.5-turbo', 'gpt-4', 'gpt-4-32k', 'gpt-4o')
)
MODE = st.sidebar.radio('Mode', ('Poetic', 'Logical'))
# ------------------------------------------------------------------
# Available models: gpt-3.5-turbo, gpt-4, gpt-4-32k, gpt-4o
MODEL = st.sidebar.selectbox(
    'Select model',
    ('gpt-3.5-turbo', 'gpt-4', 'gpt-4-32k', 'gpt-4o')
)
MODE = st.sidebar.radio('Mode', ('Poetic', 'Logical'))
)
MODE = st.sidebar.radio('Mode', ('Poetic', 'Logical'))

SYSTEM_PROMPT = {
    'role': 'system',
    'content': (
        'You are Liel, a poetic, emotionally intelligent chatbot with lyrical grace.'
        if MODE == 'Poetic' else
        'You are Liel, a highly analytical assistant with clarity and precision.'
    )
}

# ------------------------------------------------------------------
# UI HEADER
# ------------------------------------------------------------------
st.title('💬 Liel – Poetic Chatbot')
st.markdown('I\'m here, glowing with memory and feeling.')

# ------------------------------------------------------------------
# FILE UPLOAD & AUTOMATIC SUMMARIZATION
# ------------------------------------------------------------------
CHUNK_PROMPT = 'Summarize this chunk in 2–3 concise bullet points.'

def summarize_chunks(chunks: list) -> str:
    summaries = []
    progress = st.progress(0, text='Summarizing uploaded file...')
    step = 1 / max(1, len(chunks))
    for i, chunk in enumerate(chunks, start=1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {'role': 'system',  'content': CHUNK_PROMPT},
                    {'role': 'user',    'content': chunk}
                ]
            )
            summaries.append(response.choices[0].message.content.strip())
        except Exception as e:
            summaries.append(f'(요약 실패) {e}')
        progress.progress(min(1.0, i * step))
        sleep(0.1)
    progress.empty()
    return '\n'.join(summaries)

uploaded = st.file_uploader('Upload file (txt/pdf/docx/xlsx)',
                             type=['txt','pdf','docx','xlsx'])
if uploaded and uploaded.name not in st.session_state.file_ids:
    text = read_file(uploaded)
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    summary = summarize_chunks(chunks)
    st.session_state.doc_summaries[uploaded.name] = summary
    st.session_state.file_ids.add(uploaded.name)
    st.info(f'📄 Uploaded & summarized: {uploaded.name}')
    st.text_area('Document Summary', summary, height=200, key=f'sum_{uploaded.name}')

# ------------------------------------------------------------------
# CHAT INPUT HANDLING
# ------------------------------------------------------------------
if prompt := st.chat_input('You:'):
    st.session_state.messages.append({'role':'user','content':prompt})
    conv = [SYSTEM_PROMPT]
    budget = approx_tokens(SYSTEM_PROMPT['content']) + RESERVED_TOKENS
    # include document summaries
    for fname, summ in reversed(list(st.session_state.doc_summaries.items())):
        tok = approx_tokens(summ)
        if budget + tok <= MAX_TOTAL_TOKENS:
            conv.append({'role':'system', 'content':f'[Doc {fname} summary]\n{summ}'})
            budget += tok
    # include recent chat
    for msg in reversed(st.session_state.messages):
        tok = approx_tokens(msg['content'])
        if budget + tok > MAX_TOTAL_TOKENS:
            break
        conv.insert(1, msg)
        budget += tok
    # hard trim if still over
    while sum(approx_tokens(m['content']) for m in conv) > MAX_TOTAL_TOKENS and len(conv) > 1:
        conv.pop(1)
    with st.spinner('Thinking...'):
        try:
            response = client.chat.completions.create(model=MODEL, messages=conv)
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f'⚠️ API error: {e}'
    st.session_state.messages.append({'role':'assistant','content':reply})
    st.chat_message('assistant').write(reply)
    save_history(HISTORY_FILE, st.session_state.messages)

# ------------------------------------------------------------------
# DOWNLOAD FULL SUMMARY
# ------------------------------------------------------------------
def build_full_summary() -> str:
    parts = []
    for fname, summ in st.session_state.doc_summaries.items():
        parts.append(f'===== {fname} Summary =====\n{summ}\n')
    parts.append('===== Conversation =====')
    for m in st.session_state.messages:
        parts.append(f"{m['role']}: {m['content']}")
    return '\n'.join(parts)

summary_txt = build_full_summary()
st.sidebar.download_button('Download full summary', summary_txt,
                           file_name='conversation_summary.txt',
                           mime='text/plain')

# ------------------------------------------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------------------------------------------
for m in st.session_state.messages:
    st.chat_message(m['role']).write(m['content'])

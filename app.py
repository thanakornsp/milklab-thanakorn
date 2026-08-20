"""
DropExpress Smart Locker AI

Run:
    streamlit run app.py

ระบบ:
1. RAG สำหรับตอบคำถามเกี่ยวกับ DropExpress
2. Agent สำหรับทำธุรกรรม
3. Google Sheets สำหรับบันทึกธุรกรรม
4. Telegram สำหรับแจ้งเตือน
5. FAISS สำหรับค้นหา Knowledge Base
6. Gemini สำหรับ RAG และ Agent
7. บันทึก trace ลง traces.jsonl และ agent_trace.log
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from agent_harness import (
    parse_command,
    dispatch_tool,
)


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

KB_PATH = Path("locker_kb.md")

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 3


# =========================================================
# KNOWLEDGE BASE
# =========================================================

@st.cache_data
def load_knowledge_base() -> str:
    """โหลด locker_kb.md"""

    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {KB_PATH}"
        )

    text = KB_PATH.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError(
            "locker_kb.md ไม่มีข้อมูล"
        )

    return text


# =========================================================
# SPLIT MARKDOWN
# =========================================================

def split_markdown(
    text: str,
) -> list[str]:
    """แบ่ง Markdown ตามหัวข้อ ##"""

    chunks: list[str] = []
    current_lines: list[str] = []

    for line in text.splitlines():

        stripped = line.strip()

        if (
            stripped.startswith("## ")
            and current_lines
        ):
            chunk = "\n".join(
                current_lines
            ).strip()

            if chunk:
                chunks.append(chunk)

            current_lines = [line]

        else:
            current_lines.append(line)

    if current_lines:

        chunk = "\n".join(
            current_lines
        ).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    """โหลด Sentence Transformer"""

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# =========================================================
# FAISS
# =========================================================

@st.cache_resource
def build_index(
    chunks_tuple: tuple[str, ...],
):
    """สร้าง FAISS index"""

    chunks = list(
        chunks_tuple
    )

    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# RAG RETRIEVE
# =========================================================

def retrieve(
    query: str,
    chunks: list[str],
    index,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """ค้นหา Knowledge Base"""

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    actual_k = min(
        top_k,
        len(chunks),
    )

    scores, indices = index.search(
        query_embedding,
        actual_k,
    )

    results: list[dict[str, Any]] = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0:
            continue

        results.append(
            {
                "chunk_id": int(idx),
                "score": float(score),
                "text": chunks[idx],
            }
        )

    return results


# =========================================================
# RAG ANSWER
# =========================================================

def generate_rag_answer(
    question: str,
    retrieved: list[dict[str, Any]],
) -> str:
    """ให้ Gemini ตอบจาก Knowledge Base"""

    api_key = os.getenv(
        "GOOGLE_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "ไม่พบ GOOGLE_API_KEY"
        )

    from google import genai

    client = genai.Client(
        api_key=api_key
    )

    context = "\n\n".join(
        [
            (
                f"[ข้อมูล {item['chunk_id']}]\n"
                f"{item['text']}"
            )
            for item in retrieved
        ]
    )

    prompt = f"""
คุณคือ AI Assistant ของ DropExpress

DropExpress เป็นบริการตู้ล็อคเกอร์ฝากของ
และฝากส่งพัสดุอัตโนมัติ 24 ชั่วโมง

กฎสำคัญ:

1. ตอบจาก Knowledge Base เท่านั้น
2. ห้ามเดาราคา
3. ห้ามสร้างบริการใหม่
4. ห้ามสร้างเงื่อนไขใหม่
5. ถ้าไม่มีข้อมูล ให้ตอบว่า
"ขออภัย ข้อมูลส่วนนี้ยังไม่มีในระบบค่ะ 🙏"
6. ตอบภาษาไทย
7. ตอบกระชับและเข้าใจง่าย
8. ห้ามใช้ em dash

Knowledge Base:

{context}

คำถาม:

{question}
"""

    response = client.models.generate_content(
        model=os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        ),
        contents=prompt,
    )

    return (
        response.text or ""
    ).strip()


# =========================================================
# DETECT AGENT COMMAND
# =========================================================

def is_transaction_command(
    question: str,
) -> bool:
    """
    ตรวจว่าคำถามน่าจะเป็นคำสั่ง Agent หรือไม่
    """

    keywords = [
        "บันทึกฝากตู้",
        "ฝากตู้",
        "ฝากของ",
        "บันทึกการฝาก",
        "ส่งพัสดุ",
        "บันทึกส่งพัสดุ",
        "บันทึกพัสดุ",
        "วันนี้มีธุรกรรม",
        "วันนี้มียอด",
        "สรุปธุรกรรม",
        "สรุปยอด",
        "ยอดวันนี้",
        "แจ้งเตือน",
        "ส่งแจ้งเตือน",
    ]

    text = question.lower()

    return any(
        keyword in text
        for keyword in keywords
    )


# =========================================================
# AGENT
# =========================================================

def run_agent(
    question: str,
) -> str:
    """ส่งคำสั่งเข้า Agent"""

    tool_call = parse_command(
        question
    )

    tool_name = tool_call.get(
        "tool"
    )

    args = tool_call.get(
        "args",
        {},
    )

    allowed_tools = {
        "log_locker",
        "log_parcel",
        "query_transactions",
        "send_alert",
    }

    if tool_name not in allowed_tools:
        raise ValueError(
            f"Agent เลือก Tool ไม่ถูกต้อง: {tool_name}"
        )

    result = dispatch_tool(
        tool_call
    )

    return result


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="DropExpress AI",
    page_icon="📦",
    layout="centered",
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "📦 DropExpress AI"
)

st.caption(
    "Smart Locker & Logistics Assistant"
)

st.write(
    "ถามข้อมูล ใช้งานตู้ ฝากพัสดุ "
    "หรือดูธุรกรรมได้จากหน้านี้"
)


# =========================================================
# API CHECK
# =========================================================

if not os.getenv(
    "GOOGLE_API_KEY"
):

    st.error(
        "ไม่พบ GOOGLE_API_KEY"
    )

    st.info(
        "กรุณาตั้งค่า GOOGLE_API_KEY "
        "ใน Codespaces Secret"
    )

    st.stop()


# =========================================================
# LOAD RAG
# =========================================================

try:

    kb_text = load_knowledge_base()

    chunks = split_markdown(
        kb_text
    )

    if not chunks:

        st.error(
            "Knowledge Base ไม่มีข้อมูล"
        )

        st.stop()

    index = build_index(
        tuple(chunks)
    )

except Exception as exc:

    st.error(
        f"ไม่สามารถโหลด RAG ได้: {exc}"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "📦 DropExpress"
    )

    st.write(
        "Smart Locker & Logistics"
    )

    st.divider()

    st.write(
        f"Knowledge chunks: {len(chunks)}"
    )

    st.write(
        f"Embedding: {EMBEDDING_MODEL}"
    )

    st.write(
        f"Top-K: {TOP_K}"
    )

    st.divider()

    st.info(
        "RAG + AI Agent"
    )

    st.caption(
        "FAISS + Gemini + Google Sheets + Telegram"
    )

    if st.button(
        "🗑️ ล้างประวัติแชต"
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "ถามหรือสั่งงาน DropExpress ได้เลย..."
)


# =========================================================
# PROCESS MESSAGE
# =========================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    with st.chat_message(
        "assistant"
    ):

        try:

            # =================================================
            # AGENT
            # =================================================

            if is_transaction_command(
                question
            ):

                with st.spinner(
                    "กำลังดำเนินการ..."
                ):

                    result = run_agent(
                        question
                    )

                st.success(
                    result
                )

                answer = result

            # =================================================
            # RAG
            # =================================================

            else:

                with st.spinner(
                    "กำลังค้นหาข้อมูล..."
                ):

                    retrieved = retrieve(
                        question,
                        chunks,
                        index,
                        TOP_K,
                    )

                    answer = generate_rag_answer(
                        question,
                        retrieved,
                    )

                st.markdown(
                    answer
                )

            # =================================================
            # SAVE CHAT
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as exc:

            error_message = (
                f"เกิดข้อผิดพลาด: {exc}"
            )

            st.error(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )
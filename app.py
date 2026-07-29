"""MilkLab RAG Chatbot — Session 3.

Run:
    streamlit run app.py

ระบบทำงานดังนี้:
1. โหลด menu_kb.md
2. แบ่งเอกสารเป็น chunks
3. สร้าง embeddings
4. เก็บ embeddings ใน FAISS
5. Retrieve top-k
6. ส่ง context ให้ Gemini
7. บันทึก retrieve และ generate spans ลง traces.jsonl
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# =========================================================
# ตั้งค่าหลัก
# =========================================================

load_dotenv()

KB_PATH = Path("menu_kb.md")
TRACE_PATH = Path("traces.jsonl")

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)

TOP_K = 3


# =========================================================
# ฟังก์ชันแบ่งเอกสาร
# =========================================================

def split_markdown(text: str) -> list[str]:
    """แบ่ง Markdown ตามหัวข้อ ##

    แต่ละ chunk จะเก็บหัวข้อและเนื้อหาที่เกี่ยวข้องไว้ด้วยกัน
    เพื่อให้ retrieval เข้าใจบริบทของเนื้อหาได้ดีขึ้น
    """

    chunks: list[str] = []
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("## ") and current_lines:
            chunk = "\n".join(current_lines).strip()

            if chunk:
                chunks.append(chunk)

            current_lines = [line]
        else:
            current_lines.append(line)

    final_chunk = "\n".join(current_lines).strip()

    if final_chunk:
        chunks.append(final_chunk)

    # ตัด chunk ที่ว่างออก
    return [
        chunk
        for chunk in chunks
        if chunk.strip()
    ]


# =========================================================
# TODO 1+2+3
# โหลดเอกสาร แบ่ง chunk encode และสร้าง FAISS index
# =========================================================

@st.cache_resource(show_spinner="กำลังโหลดโมเดลและสร้าง FAISS index...")
def load_index(
    kb_modified_time: float,
) -> tuple[
    SentenceTransformer,
    faiss.IndexFlatIP,
    list[str],
]:
    """โหลด menu_kb.md, split, encode และสร้าง FAISS index.

    kb_modified_time มีไว้ทำให้ Streamlit สร้าง index ใหม่
    เมื่อมีการแก้ไฟล์ menu_kb.md
    """

    # ใช้เป็น cache key เท่านั้น
    del kb_modified_time

    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {KB_PATH}"
        )

    document = KB_PATH.read_text(
        encoding="utf-8",
    ).strip()

    if not document:
        raise ValueError(
            "ไฟล์ menu_kb.md ไม่มีข้อมูล"
        )

    chunks = split_markdown(document)

    if not chunks:
        raise ValueError(
            "ไม่สามารถแบ่ง knowledge base เป็น chunk ได้"
        )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

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

    # Inner Product + normalized vectors
    # ใช้เป็น cosine similarity
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return model, index, chunks


# =========================================================
# Trace / Observability
# =========================================================

def append_trace(
    span: dict[str, Any],
) -> None:
    """เพิ่ม span หนึ่งบรรทัดลง traces.jsonl."""

    with TRACE_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                span,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def create_span(
    *,
    trace_id: str,
    span_name: str,
    started_at: float,
    status: str,
    input_data: Any,
    output_data: Any,
    error: str | None = None,
) -> dict[str, Any]:
    """สร้าง span สำหรับบันทึก observability."""

    ended_at = time.time()

    return {
        "trace_id": trace_id,
        "span_id": str(uuid.uuid4()),
        "span_name": span_name,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": round(
            (ended_at - started_at) * 1000,
            2,
        ),
        "status": status,
        "input": input_data,
        "output": output_data,
        "error": error,
    }


# =========================================================
# TODO 4
# Retrieve top-k chunks
# =========================================================

def retrieve_top_k(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[str],
    *,
    trace_id: str,
    k: int = TOP_K,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Encode คำถามและค้นหา top-k chunks."""

    started_at = time.time()

    try:
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
            k,
            len(chunks),
        )

        scores, indices = index.search(
            query_embedding,
            actual_k,
        )

        results: list[dict[str, Any]] = []

        for rank, (
            chunk_index,
            score,
        ) in enumerate(
            zip(
                indices[0],
                scores[0],
            ),
            start=1,
        ):
            if chunk_index < 0:
                continue

            results.append(
                {
                    "rank": rank,
                    "chunk_id": (
                        f"chunk_{int(chunk_index):03d}"
                    ),
                    "chunk_index": int(
                        chunk_index
                    ),
                    "score": float(score),
                    "text": chunks[
                        int(chunk_index)
                    ],
                }
            )

        span = create_span(
            trace_id=trace_id,
            span_name="retrieve_top_k",
            started_at=started_at,
            status="success",
            input_data={
                "query": query,
                "k": k,
            },
            output_data={
                "results": [
                    {
                        "rank": item["rank"],
                        "chunk_id": item[
                            "chunk_id"
                        ],
                        "score": item["score"],
                    }
                    for item in results
                ]
            },
        )

        append_trace(span)

        return results, span

    except Exception as exc:
        span = create_span(
            trace_id=trace_id,
            span_name="retrieve_top_k",
            started_at=started_at,
            status="error",
            input_data={
                "query": query,
                "k": k,
            },
            output_data=None,
            error=str(exc),
        )

        append_trace(span)
        raise


# =========================================================
# สร้าง Prompt สำหรับ Gemini
# =========================================================

def build_prompt(
    query: str,
    context_chunks: list[
        dict[str, Any]
    ],
) -> str:
    """สร้าง prompt โดยบังคับให้ตอบจาก context เท่านั้น."""

    context_parts: list[str] = []

    for item in context_chunks:
        context_parts.append(
            (
                f"[{item['chunk_id']}]\n"
                f"{item['text']}"
            )
        )

    context = "\n\n".join(
        context_parts
    )

    return f"""
คุณเป็นผู้ช่วยตอบคำถามของร้าน MilkLab°

กฎสำคัญ:
1. ตอบโดยใช้เฉพาะข้อมูลใน CONTEXT เท่านั้น
2. ห้ามเดาหรือสร้างข้อมูลใหม่
3. ห้ามแต่งราคา ส่วนผสม เวลา ที่ตั้ง หรือเงื่อนไขเพิ่มเติม
4. ถ้าไม่มีคำตอบอยู่ใน CONTEXT ให้ตอบว่า:
   "ไม่พบข้อมูลนี้ในฐานความรู้ของร้าน กรุณาสอบถามพนักงาน"
5. ตอบเป็นภาษาไทย กระชับ และเข้าใจง่าย
6. ท้ายคำตอบให้ระบุ chunk_id ที่ใช้อ้างอิง

CONTEXT:
{context}

คำถามของลูกค้า:
{query}

คำตอบ:
""".strip()


# =========================================================
# TODO 5 + TODO 6
# Generate answer พร้อม span
# =========================================================

def generate_answer(
    query: str,
    context_chunks: list[
        dict[str, Any]
    ],
    *,
    trace_id: str,
) -> tuple[
    str,
    dict[str, Any],
]:
    """ส่ง query และ context ให้ Gemini."""

    started_at = time.time()

    try:
        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "ไม่พบ GEMINI_API_KEY "
                "กรุณาตั้งค่าใน Codespaces Secret "
                "หรือไฟล์ .env"
            )

        prompt = build_prompt(
            query,
            context_chunks,
        )

        client = genai.Client(
            api_key=api_key
        )

        response = (
            client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        )

        answer = (
            response.text or ""
        ).strip()

        if not answer:
            answer = (
                "ระบบไม่สามารถสร้างคำตอบได้ "
                "กรุณาลองใหม่อีกครั้ง"
            )

        span = create_span(
            trace_id=trace_id,
            span_name="generate_answer",
            started_at=started_at,
            status="success",
            input_data={
                "query": query,
                "context_chunk_ids": [
                    item["chunk_id"]
                    for item in context_chunks
                ],
                "model": GEMINI_MODEL,
            },
            output_data={
                "answer": answer,
            },
        )

        append_trace(span)

        return answer, span

    except Exception as exc:
        span = create_span(
            trace_id=trace_id,
            span_name="generate_answer",
            started_at=started_at,
            status="error",
            input_data={
                "query": query,
                "context_chunk_ids": [
                    item["chunk_id"]
                    for item in context_chunks
                ],
                "model": GEMINI_MODEL,
            },
            output_data=None,
            error=str(exc),
        )

        append_trace(span)
        raise


# =========================================================
# Streamlit Chat UI
# =========================================================

def main() -> None:
    st.set_page_config(
        page_title="MilkLab° RAG",
        page_icon="🥛",
        layout="centered",
    )

    st.title(
        "🥛 MilkLab° RAG Chatbot"
    )

    st.caption(
        "ถามเกี่ยวกับเมนู ราคา ส่วนผสม "
        "สารก่อภูมิแพ้ เวลาเปิดร้าน "
        "การจัดส่ง และ FAQ"
    )

    with st.sidebar:
        st.header(
            "สถานะระบบ"
        )

        st.write(
            f"Embedding: `{EMBEDDING_MODEL}`"
        )

        st.write(
            f"Gemini: `{GEMINI_MODEL}`"
        )

        st.write(
            f"Top-k: `{TOP_K}`"
        )

        if st.button(
            "ล้างประวัติแชต"
        ):
            st.session_state.messages = []
            st.rerun()

    try:
        kb_mtime = (
            KB_PATH.stat().st_mtime
        )

        model, index, chunks = (
            load_index(kb_mtime)
        )

        st.success(
            f"โหลดฐานความรู้สำเร็จ "
            f"{len(chunks)} chunks"
        )

    except Exception as exc:
        st.error(
            f"โหลดฐานความรู้ไม่สำเร็จ: "
            f"{exc}"
        )
        st.stop()

    if "messages" not in (
        st.session_state
    ):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "สวัสดีครับ ถามข้อมูลเกี่ยวกับ "
                    "MilkLab° ได้เลย เช่น "
                    "นมหมีฮอกไกโดราคาเท่าไร"
                ),
            }
        ]

    for message in (
        st.session_state.messages
    ):
        with st.chat_message(
            message["role"]
        ):
            st.write(
                message["content"]
            )

    prompt = st.chat_input(
        "ตัวอย่าง: ร้านเปิดกี่โมง?"
    )

    if not prompt:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message(
        "user"
    ):
        st.write(prompt)

    # หนึ่งคำถามใช้ trace_id เดียวกัน
    # ทั้ง retrieve และ generate
    trace_id = str(
        uuid.uuid4()
    )

    with st.chat_message(
        "assistant"
    ):
        try:
            with st.spinner(
                "กำลังค้นข้อมูล..."
            ):
                (
                    context,
                    retrieve_span,
                ) = retrieve_top_k(
                    prompt,
                    model,
                    index,
                    chunks,
                    trace_id=trace_id,
                    k=TOP_K,
                )

            with st.spinner(
                "กำลังสร้างคำตอบ..."
            ):
                (
                    answer,
                    generation_span,
                ) = generate_answer(
                    prompt,
                    context,
                    trace_id=trace_id,
                )

            st.write(answer)

            with st.expander(
                "Source chunks"
            ):
                for item in context:
                    st.markdown(
                        (
                            f"### อันดับ "
                            f"{item['rank']}\n"
                            f"- Chunk: "
                            f"`{item['chunk_id']}`\n"
                            f"- Similarity: "
                            f"`{item['score']:.4f}`"
                        )
                    )

                    st.write(
                        item["text"]
                    )

                    st.divider()

            # ข้อ 6 Observability
            with st.expander(
                "Trace"
            ):
                st.json(
                    {
                        "trace_id": trace_id,
                        "spans": [
                            retrieve_span,
                            generation_span,
                        ],
                    }
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as exc:
            error_message = (
                f"ระบบเกิดข้อผิดพลาด: {exc}"
            )

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )


if __name__ == "__main__":
    main()
